from __future__ import annotations

import colorsys
import functools
import random
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from nuplan.database.utils.boxes.box import BoxInterface
from nuplan.database.utils.geometry import quaternion_yaw, view_points
from nuplan.database.utils.label.label import Label
from nuplan.database.utils.plot import rainbow

# Type alias for RGBA colors.
Color = Tuple[int, int, int, int]
# Type alias for Matplotlib colors (<str> or normalized RGB tuple).
MatplotlibColor = Tuple[Union[float, str], Union[float, str], Union[float, str]]


class BoxVisibility(IntEnum):
    """Enumerates various possible requirements on the visibility of a box in an image."""

    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.


def points_in_box(box: Box3D, points: npt.NDArray[np.float64], wlh_factor: float = 1.0) -> npt.NDArray[np.float64]:
    """
    Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579.

    :param box: A Box3D instance.
    :param points: Points given as <np.float: 3, n_way_points)
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >. Mask for points in box or not.
    """
    assert points.shape[0] == 3, 'Expect 3D pts'
    assert points.ndim == 2, 'Expect 2D inputs'

    # Get the box "radius". It's actually the half of the length of longest diagonal of the cuboid.
    r = ((box.wlh / 2) ** 2).sum() ** 0.5

    w, l, h = box.wlh

    # Expand the box w, l, h and r with the wlh_factor
    w, l, h, r = w * wlh_factor, l * wlh_factor, h * wlh_factor, r * wlh_factor

    cx, cy, cz = box.center
    x, y, z = points

    pts_mask = functools.reduce(
        np.logical_and, [x >= cx - r, x <= cx + r, y >= cy - r, y <= cy + r, z >= cz - r, z <= cz + r]
    )

    pts = points[:, pts_mask]
    rot = box.orientation.inverse.rotation_matrix.astype(np.float32)

    x, y, z = rot @ pts + (rot @ -box.center.astype(np.float32)).reshape(-1, 1)
    mask = functools.reduce(
        np.logical_and,
        [
            np.logical_and(x >= -l / 2, x <= l / 2),
            np.logical_and(y >= -w / 2, y <= w / 2),
            np.logical_and(z >= -h / 2, z <= h / 2),
        ],
    )

    pts_index = np.nonzero(pts_mask)
    pts_mask[pts_index] = mask

    return pts_mask  # type: ignore


def points_in_box_bev(box: Box3D, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Checks whether points are inside the box in birds eyed view.
    :param box: Box3D instance.
    :param points: Trajectory given as <np.float: 3, n_way_points)
    :return: A boolean mask whether points are in the box in BEV world.
    """
    box = box.copy()
    points = points.copy()
    points[2, :] = box.center[2]
    return points_in_box(box, points)


def box_in_image(
    box: Box3D,
    intrinsic: npt.NDArray[np.float64],
    imsize: Tuple[float, float],
    vis_level: int = BoxVisibility.ANY,
    front: int = 2,
    min_front_th: float = 0.1,
    with_velocity: bool = False,
) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: Box3D instance.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: Image (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :param front: Which axis represents depth. Default is z-axis (2) but can be set to y-axis (1) or x-axis (0).
    :param min_front_th: Corners' depth must be greater than this threshold for a box to be in the image.
        Note that 0.1 is a number that we found to produce reasonable plots.
    :param with_velocity: If True, include the velocity endpoint as one of the corners.
    :return True if visibility condition is satisfied.
    """
    corners_3d = box.corners()

    # Add the velocity vector endpoint if it is not nan.
    if with_velocity and not np.isnan(box.velocity_endpoint).any():
        corners_3d = np.concatenate((corners_3d, box.velocity_endpoint), axis=1)

    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    # True if a corner is at least min_front_th meters in front of camera.
    in_front = corners_3d[front, :] > min_front_th
    corners_img = corners_img[:, in_front]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        # For any visibility, we don't need to check in_front here since visible is filtered with in_front.
        return any(visible)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


class Box3D(BoxInterface):
    """Simple data class representing a 3d box including, label, score and velocity."""

    MAX_LABELS = 100
    _labelmap = None
    _min_size = np.finfo(np.float32).eps  # type: ignore  # This is the smallest size within the float32 precision.
    # Modes with probabilities lower than this threshold will not be rendered.
    RENDER_MODE_PROB_THRESHOLD = 0.1

    def __init__(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        orientation: Quaternion,
        label: int = np.nan,  # type: ignore
        score: float = np.nan,
        velocity: Tuple[float, float, float] = (np.nan, np.nan, np.nan),
        angular_velocity: float = np.nan,
        payload: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
        track_token: Optional[str] = None,
        future_horizon_len_s: Optional[float] = None,
        future_interval_s: Optional[float] = None,
        future_centers: Optional[List[List[Tuple[float, float, float]]]] = None,
        future_orientations: Optional[List[List[Quaternion]]] = None,
        mode_probs: Optional[List[float]] = None,
    ) -> None:
        """
        The convention is that: x points forward, y to the left, z up when this box is initialized with an orientation
        of zero.
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box3D orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box3D velocity in x, y, z direction.
        :param angular_velocity: Box3D angular velocity in yaw direction.
        :param payload: Box3D payload, optional. For example, can be used to denote category name or provide boolean
            data regarding whether the box trajectory goes off the driveable area. The format should be a dictionary
            so that different types of metadata can be stored here, e.g., payload['category_name'] and
            payload['timestamp_2_on_road_bool'].
        :param token: Unique token (optional). Usually DB annotation token. In NuPlanDB, 3D annotations are present in
            the LidarBox table, in which case the token provided corresponds to the LidarBox token.
        :param track_token: Track token in the "track" table that corresponds to a particular box.
        :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
        :param future_interval_s: Timestamp interval of the future waypoints in seconds.
        :param future_centers: List of future center coordinates given as (x, y, z), where the list indices increase
            with time and are spaced apart at the specified intervals. If the box is missing at a future timestamp, then
            the future center coordinates at the corresponding list index will have the format (np.nan, np.nan, np.nan)
        :param future_orientations: List of future Box3D orientations, where the list indices increase with time and
            are spaced apart at the specified intervals. If the box is missing at a future timestamp, then
            the future orientation at the corresponding list index will be represented as None.
        :param mode_probs: Mode probabilities.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert len(velocity) == 3
        assert type(orientation) == Quaternion

        # Require width, length, and height to be positive and more than 0
        assert size[0] > self._min_size, "Error: box Width must be larger than {} cm".format(100 * self._min_size)
        assert size[1] > self._min_size, "Error: box Length must be larger than {} cm".format(100 * self._min_size)
        assert size[2] > self._min_size, "Error: box Height must be larger than {} cm".format(100 * self._min_size)

        # Require the box volume to be above float32 precision
        assert size[0] * size[1] * size[2] > self._min_size, 'Invalid box volume'

        self.center = np.array(center, dtype=float)  # type: ignore
        self.size = size
        self.wlh = np.array(size, dtype=float)  # type: ignore
        # Require an explicit copy to ensure boxes can be freely manipulated. Need to use Quaternion's private method
        # since the copy mentioned in the Pyquaternion documentation does not work as intended.
        self.orientation = orientation.__copy__()
        self._label = int(label) if not np.isnan(label) else label
        self._score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity, dtype=float)  # type: ignore
        self.angular_velocity = float(angular_velocity) if not np.isnan(angular_velocity) else angular_velocity
        self.payload = payload if payload is not None else {}
        assert type(self.payload) == dict, "Error: box payload is not a dict"

        self.token = token
        self._color = None
        self.track_token = track_token

        self.init_trajectory_fields(
            future_horizon_len_s,
            future_interval_s,
            future_centers,
            future_orientations,
            mode_probs,
        )

    @classmethod
    def set_labelmap(cls, labelmap: Dict[int, Label]) -> None:
        """
        :param labelmap: {id: label}. Map from label id to Label.
        """
        cls._labelmap = labelmap

    @property
    def color(self) -> Color:
        """RGBA color of Box3D."""
        if self._color is None:
            self._set_color()

        return self._color  # type: ignore

    @property
    def width(self) -> float:
        """Width of the box."""
        return float(self.wlh[0])

    @width.setter
    def width(self, width: float) -> None:
        """Implemented. See interface."""
        self.wlh[0] = width

    @property
    def length(self) -> float:
        """Length of the box."""
        return float(self.wlh[1])

    @length.setter
    def length(self, length: float) -> None:
        """Implemented. See interface."""
        self.wlh[1] = length

    @property
    def height(self) -> float:
        """Height of the box."""
        return float(self.wlh[2])

    @height.setter
    def height(self, height: float) -> None:
        """Implemented. See interface."""
        self.wlh[2] = height

    @property
    def yaw(self) -> float:
        """Yaw of the box."""
        return quaternion_yaw(self.orientation)  # type: ignore

    @property
    def distance_plane(self) -> float:
        """
        The euclidean distance of the box center from the z-axis passing through the origin of the coordinate system
        (sensor/world). Refer to the axial/radial distance in a cylindrical coordinate system:
        https://en.wikipedia.org/wiki/Cylindrical_coordinate_system.
        """
        return float((self.center[0] ** 2 + self.center[1] ** 2) ** 0.5)

    @property
    def distance_3d(self) -> float:
        """
        The euclidean distance of the box center from the origin of the coordinate system (sensor/world). Refer to the
        radial distance in a spherical coordinate system: https://en.wikipedia.org/wiki/Spherical_coordinate_system.
        """
        return float((self.center[0] ** 2 + self.center[1] ** 2 + self.center[2] ** 2) ** 0.5)

    def init_trajectory_fields(
        self,
        future_horizon_len_s: Optional[float] = None,
        future_interval_s: Optional[float] = None,
        future_centers: Optional[List[List[Tuple[float, float, float]]]] = None,
        future_orientations: Optional[List[List[Quaternion]]] = None,
        mode_probs: Optional[List[float]] = None,
    ) -> None:
        """
        Checks that values for future horizon length, interval length, future orientations and future centers are either
        all provided or all None. Check that future centers and future orientations are the expected length, if
        applicable.
        :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
        :param future_interval_s: Timestamp interval of the future waypoints in seconds.
        :param future_centers: List of future center coordinates given as (x, y, z), where the list indices increase
            with time and are spaced apart at the specified intervals. If the box is missing at a future timestamp, then
            the future center coordinates at the corresponding list index will have the format (np.nan, np.nan, np.nan)
        :param future_orientations: List of future Box3D orientations, where the list indices increase with time and
            are spaced apart at the specified intervals. If the box is missing at a future timestamp, then
            the future orientation at the corresponding list index will be represented as None.
        :param mode_probs: Mode probabilities.
        """
        if future_centers is None:
            assert future_horizon_len_s is None
            assert future_interval_s is None
            assert future_orientations is None
            assert mode_probs is None
            self.future_horizon_len_s = None
            self.future_interval_s = None
            self.future_centers = None
            self.future_orientations = None
            self.mode_probs = None
            self.num_modes = None
            self.num_future_timesteps = None
            return

        assert future_horizon_len_s is not None
        assert future_interval_s is not None
        assert future_orientations is not None
        assert mode_probs is not None
        self.future_horizon_len_s = future_horizon_len_s
        self.future_interval_s = future_interval_s
        self.future_centers = np.array(future_centers, dtype=float)
        self.future_orientations = future_orientations
        self.mode_probs = np.array(mode_probs, dtype=float)

        # Check number of modes match.
        assert self.future_centers.ndim == 3  # type: ignore
        if not self.mode_probs.shape[0] == self.future_centers.shape[0] == len(self.future_orientations):  # type: ignore
            raise ValueError(
                f"Future parameters have different number of modes:\n"  # type: ignore
                f"self.mode_probs.shape: {self.mode_probs.shape}\n"
                f"self.future_centers.shape: {self.future_centers.shape}\n"
                f"len(self.future_orientations): {len(self.future_orientations)}"
            )
        self.num_modes = self.mode_probs.shape[0]  # type: ignore

        # Check number of timesteps match.
        if self.future_centers.shape[1] != len(self.future_orientations[0]):  # type: ignore
            raise ValueError(
                f"Future parameters have different number of timesteps:\n"  # type: ignore
                f"self.future_centers.shape: {self.future_centers.shape}\n"
                f"len(self.future_orientations[0]): {len(self.future_orientations[0])}"
            )
        self.num_future_timesteps = self.future_centers.shape[1]  # type: ignore

        if self.future_horizon_len_s != self.future_interval_s * self.num_future_timesteps:
            raise ValueError(
                f"Future horizon length ({self.future_horizon_len_s}) should equal to "
                f"future interval ({self.future_interval_s}) times number of timesteps ({self.num_future_timesteps})."
            )

    def _set_color(self) -> None:
        """Sets color based on label."""
        if self._labelmap is None or self.label not in self._labelmap:
            if self.label is None or np.isnan(self.label):
                self._color = (255, 61, 99, 0)  # type: ignore  # Set color to Red when label is not set.
            else:
                # When label is set, but there is no labelmap, return color from fixed permutation of rainbow method.
                # Usually first three classes are vehicle, bike and pedestrian so reserve the standard colors for these.
                # orange, Red, blue
                fixed_colors = [(255, 61, 99, 0), (255, 158, 0, 0), (0, 0, 230, 0)]
                colors = [el + (255,) for el in rainbow(self.MAX_LABELS - 3)]
                random.Random(1).shuffle(colors)
                colors = fixed_colors + colors
                self._color = colors[self.label % self.MAX_LABELS]  # type: ignore
        else:
            self._color = self._labelmap[self.label].color

    @property
    def name(self) -> str:
        """Name of Box3D."""
        if self._labelmap is None or self.label is np.nan:
            return 'not_set'
        elif self.label not in self._labelmap:
            return 'unknown'
        else:
            return self._labelmap[self.label].name  # type: ignore

    @property
    def label(self) -> int:
        """Implemented. See interface."""
        return self._label

    @label.setter
    def label(self, label: int) -> None:
        """Implemented. See interface."""
        self._label = label

    @property
    def score(self) -> float:
        """Implemented. See interface."""
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        """Implemented. See interface."""
        self._score = score

    @property
    def has_future_waypoints(self) -> bool:
        """Whether this box has future waypoints."""
        return self.future_centers is not None

    def equate_orientations(self, other: object) -> bool:
        """
        Compare orientations of two Box3D Objects.
        :param other: The other Box3D object.
        :return: True if orientations of both objects are the same, otherwise False.
        """
        if (self.future_orientations is None) != (other.future_orientations is None):  # type: ignore
            return False
        if self.future_orientations is not None and other.future_orientations is not None:  # type: ignore
            for mode_idx in range(self.num_modes):  # type: ignore
                for horizon_idx in range(self.num_future_timesteps):  # type: ignore
                    self_future_orientation = self.future_orientations[mode_idx][horizon_idx]
                    other_future_orientation = other.future_orientations[mode_idx][horizon_idx]  # type: ignore
                    if (self_future_orientation is None) != (other_future_orientation is None):
                        return False
                    if self_future_orientation is not None and other_future_orientation is not None:
                        if not np.allclose(
                            self.future_orientations[mode_idx][horizon_idx].rotation_matrix,
                            other.future_orientations[mode_idx][horizon_idx].rotation_matrix,  # type: ignore
                            atol=1e-04,
                        ):
                            return False
        return True

    def __eq__(self, other: object) -> bool:
        """
        Compares the two Box3D object are the same.
        :param other: The other Box3D object.
        :return: True if both objects are the same, otherwise False.
        """
        if not isinstance(other, Box3D):
            return NotImplemented

        center = np.allclose(self.center, other.center, atol=1e-04)
        wlh = np.allclose(self.wlh, other.wlh, atol=1e-04)
        orientation = np.allclose(self.orientation.rotation_matrix, other.orientation.rotation_matrix, atol=1e-04)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = np.allclose(self.velocity, other.velocity, atol=1e-04) or (
            np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))
        )
        angular_vel = np.isclose(self.angular_velocity, other.angular_velocity, atol=1e-04) or (
            np.isnan(self.angular_velocity) and np.isnan(other.angular_velocity)
        )
        payload = self.payload == other.payload

        if not (center and wlh and orientation and label and score and vel and angular_vel and payload):
            return False

        if self.future_horizon_len_s != other.future_horizon_len_s:
            return False
        if self.future_interval_s != other.future_interval_s:
            return False
        if self.num_future_timesteps != other.num_future_timesteps:
            return False
        if self.num_modes != other.num_modes:
            return False
        if (self.future_centers is None) != (other.future_centers is None):
            return False
        if self.future_centers is not None and other.future_centers is not None:
            if not np.array_equal(np.isnan(self.future_centers), np.isnan(other.future_centers)):
                return False
            if not np.allclose(
                self.future_centers[~np.isnan(self.future_centers)],
                other.future_centers[~np.isnan(other.future_centers)],
                atol=1e-04,
            ):
                return False

        if not self.equate_orientations(other):
            return False

        if (self.mode_probs is None) != (other.mode_probs is None):
            return False
        if self.mode_probs is not None and other.mode_probs is not None:
            if not np.allclose(self.mode_probs, other.mode_probs, atol=1e-04):
                return False

        return True

    def __repr__(self) -> str:
        """
        Represent a box using a string.
        :return: A string to represent a box.
        """
        arguments = 'center={}, size={}, orientation={}'.format(
            tuple(self.center), tuple(self.wlh), self.orientation.__repr__()
        )
        if not np.isnan(self.label):
            arguments += ', label={}'.format(self.label)
        if not np.isnan(self.score):
            arguments += ', score={}'.format(self.score)
        if not all(np.isnan(self.velocity)):
            arguments += ', velocity={}'.format(tuple(self.velocity))
        if not np.isnan(self.angular_velocity):
            arguments += ', angular_velocity={}'.format(self.angular_velocity)
        if self.payload is not None:
            arguments += ', payload=\'{}\''.format(self.payload)
        if self.token is not None:
            arguments += ', token=\'{}\''.format(self.token)
        if self.track_token is not None:
            arguments += ', track_token=\'{}\''.format(self.track_token)
        if self.future_horizon_len_s is not None:
            arguments += ', future_horizon_len_s=\'{}\''.format(self.future_horizon_len_s)
        if self.future_interval_s is not None:
            arguments += ', future_interval_s=\'{}\''.format(self.future_interval_s)
        if self.future_centers is not None:
            arguments += ', future_centers=\'{}\''.format(self.future_centers)
        if self.future_orientations is not None:
            arguments += ', future_orientations=\'{}\''.format(self.future_orientations)
        if self.mode_probs is not None:
            arguments += ', mode_probs=\'{}\''.format(self.mode_probs)

        return 'Box3D({})'.format(arguments)

    def serialize(self) -> Dict[str, Any]:
        """
        Implemented. See interface.
        :return: Dict of field name to field values.
        """
        future_orientations_serialized = (
            [
                [
                    orientation.elements.tolist() if orientation is not None else None
                    for orientation in future_orientations_of_mode
                ]
                for future_orientations_of_mode in self.future_orientations
            ]
            if self.future_orientations is not None
            else None
        )

        return {
            'center': self.center.tolist(),
            'wlh': self.wlh.tolist(),
            'orientation': self.orientation.elements.tolist(),
            'label': self.label,
            'score': self.score,
            'velocity': self.velocity.tolist(),
            'angular_velocity': self.angular_velocity,
            'payload': self.payload,
            'token': self.token,
            'track_token': self.track_token,
            'future_horizon_len_s': self.future_horizon_len_s,
            'future_interval_s': self.future_interval_s,
            'future_centers': self.future_centers.tolist() if self.future_centers is not None else None,  # type: ignore
            'future_orientations': future_orientations_serialized,
            'mode_probs': self.mode_probs.tolist() if self.mode_probs is not None else None,  # type: ignore
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Box3D:
        """
        Implemented. See interface.
        :param data: Output from serialize.
        :return: Deserialized Box3D.
        """
        if type(data) is dict:
            future_orientations = (
                [
                    [
                        Quaternion(orientation) if orientation is not None else None
                        for orientation in orientations_of_mode
                    ]
                    for orientations_of_mode in data['future_orientations']
                ]
                if data['future_orientations'] is not None
                else None
            )
            return Box3D(
                data['center'],
                data['wlh'],
                Quaternion(data['orientation']),
                label=data['label'],
                score=data['score'],
                velocity=data['velocity'],
                angular_velocity=data['angular_velocity'],
                payload=data['payload'],
                token=data['token'],
                track_token=data['track_token'],
                future_horizon_len_s=data['future_horizon_len_s'],
                future_interval_s=data['future_interval_s'],
                future_centers=data['future_centers'],
                future_orientations=future_orientations,
                mode_probs=data['mode_probs'],
            )
        else:
            raise TypeError("Type of data should be a dictionary.")

    @classmethod
    def arbitrary_box(cls) -> Box3D:
        """Instantiates an arbitrary box."""
        return Box3D(
            center=(1.1, 2.2, 3.3),
            size=(2.2, 5.5, 3.1),
            orientation=Quaternion(1, 2, 3, 4),
            label=1,
            score=0.5,
            velocity=(1.1, 2.3, 3.3),
            angular_velocity=0.314,
            payload={'def': 'hij'},
            token='abc',
            track_token='wxy',
        )

    @classmethod
    def make_random(cls) -> Box3D:
        """
        Instantiates a random box.
        :return: Box3D instance.
        """
        # TODO: add the other fields.
        center = random.sample(range(50), 3)
        size = random.sample(range(1, 50), 3)
        quaternion = Quaternion(random.sample(range(10), 4))
        label = random.choice(range(cls.MAX_LABELS))
        score = random.uniform(0, 1)
        velocity = tuple(random.uniform(0, 10) for _ in range(3))
        angular_velocity = np.random.uniform(-np.pi, np.pi)

        return Box3D(
            center=center,  # type: ignore
            size=size,  # type: ignore
            orientation=quaternion,
            label=label,
            score=score,
            velocity=velocity,  # type: ignore
            angular_velocity=angular_velocity,
        )

    def copy(self) -> Box3D:
        """
        Create a copy of self.
        :return: Box3D instance.
        """
        return Box3D(
            center=self.center,  # type: ignore
            size=self.wlh,  # type: ignore
            orientation=self.orientation,
            label=self.label,
            score=self.score,
            velocity=self.velocity,  # type: ignore
            angular_velocity=self.angular_velocity,
            payload=self.payload,
            token=self.token,
            track_token=self.track_token,
            future_horizon_len_s=self.future_horizon_len_s,
            future_interval_s=self.future_interval_s,
            future_centers=self.future_centers,
            future_orientations=self.future_orientations,
            mode_probs=self.mode_probs,
        )

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """
        Returns a rotation matrix.
        :return: <np.float: (3, 3)>.
        """
        return self.orientation.rotation_matrix  # type: ignore

    def translate(self, x: npt.NDArray[np.float64]) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3>. Translation in x, y, z direction.
        """
        self.center += x

        if self.future_centers is not None:
            assert x.ndim == 1
            assert x.shape[-1] == self.future_centers.shape[-1]
            # Broadcast to the last dimension.
            self.future_centers += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates a box.
        :param quaternion: Rotation to apply.
        """
        self.orientation = quaternion * self.orientation
        rotation_matrix = quaternion.rotation_matrix
        self.center = np.dot(rotation_matrix, self.center)
        self.velocity = np.dot(rotation_matrix, self.velocity)

        if self.future_centers is not None:
            for mode_idx in range(self.num_modes):
                for horizon_idx in range(self.num_future_timesteps):
                    self.future_centers[mode_idx][horizon_idx] = np.dot(
                        rotation_matrix, self.future_centers[mode_idx][horizon_idx]
                    )

        if self.future_orientations is not None:
            for mode_idx in range(self.num_modes):  # type: ignore
                for horizon_idx in range(self.num_future_timesteps):  # type: ignore
                    if self.future_orientations[mode_idx][horizon_idx] is None:
                        continue
                    self.future_orientations[mode_idx][horizon_idx] = (
                        quaternion * self.future_orientations[mode_idx][horizon_idx]
                    )

    def transform(self, trans_matrix: npt.NDArray[np.float64]) -> None:
        """
        Applies a transformation matrix to the box
        :param trans_matrix: <np.float: 4, 4>. Homogeneous transformation matrix.
        """
        self.rotate(Quaternion(matrix=trans_matrix[:3, :3]))
        self.translate(trans_matrix[:3, 3])

    def scale(self, s: Tuple[float, float, float]) -> None:
        """
        Scales the box coordinate system.
        :param s: Scale parameter in x, y, z direction.
        """
        scale = np.asarray(s)  # type: ignore
        assert len(scale) == 3
        self.center *= scale
        self.wlh *= scale
        self.velocity *= scale

        if self.future_centers is not None:
            assert scale.ndim == 1
            assert scale.shape[-1] == self.future_centers.shape[-1]
            # Broadcast to the last dimension.
            self.future_centers *= scale

    def xflip(self) -> None:
        """Flip the box along the X-axis."""
        self.center[0] *= -1
        self.velocity[0] *= -1
        self.angular_velocity *= -1

        if self.future_centers is not None:
            self.future_centers[:, :, 0] *= -1

        # Calculate required orientation flip.
        current_yaw = quaternion_yaw(self.orientation)
        final_yaw = -current_yaw + np.pi
        self.orientation = Quaternion(axis=(0, 0, 1), angle=final_yaw)

        if self.future_orientations is not None:
            for mode_idx in range(self.num_modes):  # type: ignore
                for horizon_idx in range(self.num_future_timesteps):  # type: ignore
                    orientation = self.future_orientations[mode_idx][horizon_idx]
                    if orientation is None:
                        continue
                    current_yaw = quaternion_yaw(orientation)
                    final_yaw = -current_yaw + np.pi
                    self.future_orientations[mode_idx][horizon_idx] = Quaternion(axis=(0, 0, 1), angle=final_yaw)

    def yflip(self) -> None:
        """Flip the box along the Y-axis."""
        self.center[1] *= -1
        self.velocity[1] *= -1
        self.angular_velocity *= -1

        if self.future_centers is not None:
            self.future_centers[:, :, 1] *= -1

        # Calculate required orientation flip.
        current_yaw = quaternion_yaw(self.orientation)
        final_yaw = -current_yaw
        self.orientation = Quaternion(axis=(0, 0, 1), angle=final_yaw)

        if self.future_orientations is not None:
            for mode_idx in range(self.num_modes):  # type: ignore
                for horizon_idx in range(self.num_future_timesteps):  # type: ignore
                    orientation = self.future_orientations[mode_idx][horizon_idx]
                    if orientation is None:
                        continue
                    current_yaw = quaternion_yaw(orientation)
                    final_yaw = -current_yaw
                    self.future_orientations[mode_idx][horizon_idx] = Quaternion(axis=(0, 0, 1), angle=final_yaw)

    def corners(self, wlh_factor: float = 1.0) -> npt.NDArray[np.float64]:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w: float = self.wlh[0] * wlh_factor
        l: float = self.wlh[1] * wlh_factor
        h: float = self.wlh[2] * wlh_factor
        # We need to make center and rotation_matrix hashable to use @functools.lru_cached().
        center = tuple(self.center.flatten())
        rotation_matrix = tuple(self.rotation_matrix.flatten())
        return self._calc_corners(w, l, h, center, rotation_matrix)

    @property
    def front_corners(self) -> npt.NDArray[np.float64]:
        """
        Returns the four corners of the front face of the box. First two are on top face while the last two are on the
        bottom face.
        :return: <np.float: 3, 4>. Front corners.
        """
        return self.corners()[:, :4]  # type: ignore

    @property
    def rear_corners(self) -> npt.NDArray[np.float64]:
        """
        Returns the four corners of the rear face of the box. First two are on top face while the last two are on the
        bottom face.
        :return: <np.float: 3, 4>. Rear corners.
        """
        return self.corners()[:, 4:]  # type: ignore

    @property
    def bottom_corners(self) -> npt.NDArray[np.float64]:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]  # type: ignore

    @property
    def center_bottom_forward(self) -> npt.NDArray[np.float64]:
        """
        Returns the coordinate of the following point: the center of the intersection of the bottom and forward faces
        of the box.
        :return: <np.float: 3, 1>.
        """
        return np.expand_dims(np.mean(self.corners().T[2:4], axis=0), 0).T

    @property
    def front_center(self) -> npt.NDArray[np.float64]:
        """
        Returns the coordinate of the center of the front face of the box.
        :return: <np.float: 3>.
        """
        return np.mean(self.front_corners, axis=1)  # type: ignore

    @property
    def rear_center(self) -> npt.NDArray[np.float64]:
        """
        Returns the coordinate of the center of the rear face of the box.
        :return: <np.float: 3>.
        """
        return np.mean(self.rear_corners, axis=1)  # type: ignore

    @property
    def bottom_center(self) -> npt.NDArray[np.float64]:
        """
        Returns the coordinate of the bottom face center.
        :return: <np.float: 3>.
        """
        return np.mean(self.bottom_corners, axis=1)  # type: ignore

    @property
    def velocity_endpoint(self) -> npt.NDArray[np.float64]:
        """
        Extends the velocity vector from the front bottom center.
        :return: <np.float: 3, 1>.
        """
        return self.center_bottom_forward + np.expand_dims(self.velocity.T, axis=1)

    def get_future_horizon_idx(self, future_horizon_s: float) -> int:
        """
        Gets the index of a future horizon.
        :param future_horizon_s: Future horizon in seconds.
        :return: The index of the future horizon.
        """
        if self.future_horizon_len_s is None or self.future_interval_s is None:
            raise ValueError(
                f"Future horizon information is not available. Invalid variable values:\n"
                f"future_horizon_len_s={self.future_horizon_len_s}\nfuture_interval_s={self.future_interval_s}."
            )

        if not 0.0 < future_horizon_s <= self.future_horizon_len_s:
            raise ValueError(f"Future horizon ({future_horizon_s}) should be in (0, {self.future_horizon_len_s}].")
        horizon_idx = round(future_horizon_s / self.future_interval_s - 1, 1)
        if not horizon_idx.is_integer():
            raise ValueError(
                f"Future horizon ({future_horizon_s}) divided by future interval ({self.future_interval_s}) "
                "is not an integer."
            )
        horizon_idx = int(horizon_idx)
        assert 0 <= horizon_idx < self.num_future_timesteps  # type: ignore
        return horizon_idx

    def get_all_future_horizons_s(self) -> List[float]:
        """
        Gets the list of all future horizons.
        :return: The list of all future horizons.
        """
        return [
            round((horizon_idx + 1) * self.future_interval_s, 2)  # type: ignore
            for horizon_idx in range(self.num_future_timesteps)  # type: ignore
        ]

    def get_future_center_at_horizon(self, future_horizon_s: float) -> npt.NDArray[np.float64]:
        """
        Gets future center of the highest probability trajectory at a given horizon.
        :param future_horizon_s: Future horizon in seconds.
        :return: Future center at the given horizon.
        """
        if self.future_centers is None:
            raise ValueError("Future center is not available.")

        highest_prob_mode_idx = self.get_highest_prob_mode_idx()
        horizon_idx = self.get_future_horizon_idx(future_horizon_s)
        return self.future_centers[highest_prob_mode_idx, horizon_idx]

    def get_future_centers_at_horizons(self, future_horizons_s: List[float]) -> npt.NDArray[np.float64]:
        """
        Gets future centers at the given horizons.
        :param future_horizons_s: Future horizons in seconds.
        :return: Future centers at the given horizons.
        """
        if self.future_centers is None:
            raise ValueError("Future center is not available.")

        highest_prob_mode_idx = self.get_highest_prob_mode_idx()
        horizon_indices = [self.get_future_horizon_idx(future_horizon_s) for future_horizon_s in future_horizons_s]
        return self.future_centers[highest_prob_mode_idx, horizon_indices]

    def get_future_orientation_at_horizon(self, future_horizon_s: float) -> Quaternion:
        """
        Gets future orientation of the highest probability trajectory at a given horizon.
        :param future_horizon_s: Future horizon in seconds.
        :return: Future orientation at the given horizon.
        """
        if self.future_orientations is None:
            raise ValueError("Future orientation is not available.")

        highest_prob_mode_idx = self.get_highest_prob_mode_idx()
        horizon_idx = self.get_future_horizon_idx(future_horizon_s)
        return self.future_orientations[highest_prob_mode_idx][horizon_idx]

    def get_future_orientations_at_horizons(self, future_horizons_s: List[float]) -> List[Quaternion]:
        """
        Gets future orientation of the highest probability trajectory at the given horizons.
        :param future_horizons_s: Future horizons in seconds.
        :return: Future orientations at the given horizons.
        """
        if self.future_orientations is None:
            raise ValueError("Future orientation is not available.")

        highest_prob_mode_idx = self.get_highest_prob_mode_idx()
        horizon_indices = [self.get_future_horizon_idx(future_horizon_s) for future_horizon_s in future_horizons_s]
        return [self.future_orientations[highest_prob_mode_idx][horizon_idx] for horizon_idx in horizon_indices]

    def get_topk_future_center_at_horizon(self, future_horizon_s: float, topk: int) -> npt.NDArray[np.float64]:
        """
        Gets top-k future centers at a given horizon.
        :param future_horizon_s: Future horizon in seconds.
        :param topk: The number of top-k modes.
        :return: Future center at the given horizon.
        """
        if self.future_centers is None:
            raise ValueError("Future centers are not available.")

        topk_mode_indices = self.get_topk_mode_indices(topk)
        horizon_idx = self.get_future_horizon_idx(future_horizon_s)

        return self.future_centers[topk_mode_indices, horizon_idx]

    def get_topk_future_orientation_at_horizon(self, future_horizon_s: float, topk: int) -> List[Quaternion]:
        """
        Gets top-k future orientations at a given horizon.
        :param future_horizon_s: Future horizon in seconds.
        :param topk: The number of top-k modes.
        :return: Future orientation at the given horizon.
        """
        if self.future_orientations is None:
            raise ValueError("Future orientations are not available.")

        topk_mode_indices = self.get_topk_mode_indices(topk)
        horizon_idx = self.get_future_horizon_idx(future_horizon_s)
        return [self.future_orientations[mode_idx][horizon_idx] for mode_idx in topk_mode_indices]

    def get_topk_mode_indices(self, topk: int) -> List[int]:
        """
        Gets the indices for the top-k highest probability modes.
        :param topk: Number of top-k modes.
        :return: The list of top-k highest probability mode indices.
        """
        if self.mode_probs is None:
            raise ValueError("Mode probabilities are not available.")

        return self.mode_probs.argsort()[::-1][:topk]

    def get_highest_prob_mode_idx(self) -> int:
        """
        Gets the index of the highest probability mode.
        :return: The index of the highest probability mode.
        """
        return self.get_topk_mode_indices(1)[0]

    def draw_line(
        self,
        canvas: Union[plt.Axes, npt.NDArray[np.uint8]],
        from_x: float,
        to_x: float,
        from_y: float,
        to_y: float,
        color: Tuple[Union[float, str], Union[float, str], Union[float, str]],
        linewidth: float,
        marker: Optional[str] = None,
        alpha: float = 1.0,
    ) -> None:
        """
        Draws a line on a matplotlib/cv2 canvas.
        :param canvas: <matplotlib.pyplot.axis> OR <np.array: width, height, 3>.
        Axis/Image onto which the box should be drawn.
        :param from_x: The start x coordinates of vertices.
        :param to_x: The end x coordinates of vertices.
        :param from_y: The start y coordinates of vertices.
        :param to_y: The end y coordinates of vertices.
        :param color: The color used to draw line.
        :param linewidth: Width in pixel of the box sides.
        :param marker: Marker style string to draw line.
        :param alpha: The degree of transparency (or opacity) of a color.
        """
        # Draw a line on a matplotlib/cv2 canvas. Note that marker is not used in cv2.
        if isinstance(canvas, np.ndarray):
            color_int = tuple(int(c * 255) for c in color)
            cv2.line(canvas, (int(from_x), int(from_y)), (int(to_x), int(to_y)), color_int[::-1], linewidth)
        else:
            canvas.plot([from_x, to_x], [from_y, to_y], color=color, linewidth=linewidth, marker=marker, alpha=alpha)

    def draw_rect(
        self,
        canvas: Union[plt.Axes, npt.NDArray[np.uint8]],
        selected_corners: npt.NDArray[np.float64],
        color: Tuple[float, float, float],
        linewidth: float,
    ) -> None:
        """
        Draws a rectangle on a matplotlib/cv2 canvas.
        :param canvas: <matplotlib.pyplot.axis> OR <np.array: width, height, 3>.
        Axis/Image onto which the box should be drawn.
        :param selected_corners: The selected corners for a rectangle.
        :param color: The color used to draw rectangle.
        :param linewidth: Width in pixel of the box sides.
        """
        prev = selected_corners[-1]
        for corner in selected_corners:
            self.draw_line(canvas, prev[0], corner[0], prev[1], corner[1], color=color, linewidth=linewidth)
            prev = corner

    def draw_text(self, canvas: Union[plt.Axes, npt.NDArray[np.uint8]], x: float, y: float, text: str) -> None:
        """
        Draws text on a matplotlib/cv2 canvas.
        :param canvas: <matplotlib.pyplot.axis> OR <np.array: width, height, 3>.
        Axis/Image onto which the box should be drawn.
        :param x: The x coordinates of vertices.
        :param y: The y coordinates of vertices.
        :param text: The text to draw.
        """
        if isinstance(canvas, np.ndarray):
            cv2.putText(canvas, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            canvas.text(x, y, text)

    def render(
        self,
        canvas: Union[plt.Axes, npt.NDArray[np.uint8]],
        view: npt.NDArray[np.float64] = np.eye(3),
        normalize: bool = False,
        colors: Tuple[MatplotlibColor, MatplotlibColor, MatplotlibColor] = None,  # type: ignore
        linewidth: float = 2,
        marker: str = 'o',
        with_direction: bool = True,
        with_velocity: bool = False,
        with_label: bool = False,
    ) -> None:
        """
        Renders the box. Canvas can be either a Matplotlib axis or a numpy array image (using cv2).
        :param canvas: <matplotlib.pyplot.axis> OR <np.array: width, height, 3>.
            Axis/Image onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            rear/top and bottom.
        :param linewidth: Width in pixel of the box sides.
        :param marker: Marker style string to draw line.
        :param with_direction: Whether to draw a line indicating box direction.
        :param with_velocity: Whether to draw a line indicating box velocity.
        :param with_label: Whether to render the label.
        """
        # Get the box corners.
        corners = self.corners()

        # Points that are behind the camera need to be flipped along the depth/z axis for the projection to work.
        sel = corners[2, :] < 0
        corners[2, sel] *= -1
        corners = view_points(corners, view, normalize=normalize)[:2, :]

        # Set colors if it's None.
        if colors is None:
            color = tuple(c / 255 for c in self.color[:3])
            colors = (color, color, 'k')

        # Replace string colors with numbers.
        colors = tuple(matplotlib.colors.to_rgb(c) if isinstance(c, str) else c for c in colors)  # type: ignore

        # Draw the bottom sides first to convey depth ordering.
        for i in [2, 3]:
            self.draw_line(
                canvas,
                corners.T[i][0],
                corners.T[i + 4][0],
                corners.T[i][1],
                corners.T[i + 4][1],
                color=colors[2],
                linewidth=linewidth,
            )

        # Draw the top sides.
        for i in [0, 1]:
            self.draw_line(
                canvas,
                corners.T[i][0],
                corners.T[i + 4][0],
                corners.T[i][1],
                corners.T[i + 4][1],
                color=colors[1],
                linewidth=linewidth,
            )

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d).
        self.draw_rect(canvas, corners.T[:4], colors[0], linewidth)  # type: ignore
        self.draw_rect(canvas, corners.T[4:], colors[1], linewidth)  # type: ignore

        # Draw line indicating the direction.
        if with_direction:
            center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
            center_bottom_forward = np.mean(corners.T[2:4], axis=0)
            self.draw_line(
                canvas,
                center_bottom[0],
                center_bottom_forward[0],
                center_bottom[1],
                center_bottom_forward[1],
                color=colors[1],
                linewidth=linewidth,
            )

        # Draw line to represent the velocity.
        if with_velocity and not any(np.isnan(self.velocity)):
            center_bottom_forward = np.mean(corners.T[2:4], axis=0)
            velocity_end = view_points(self.velocity_endpoint, view, normalize=normalize)[:2, 0]
            self.draw_line(
                canvas,
                center_bottom_forward[0],
                velocity_end[0],
                center_bottom_forward[1],
                velocity_end[1],
                color=colors[1],
                linewidth=linewidth * 2,
                marker='o',
            )

        # Write the label in the middle of the box.
        if with_label:
            org_center = np.expand_dims(self.center, axis=0).T  # type: ignore
            proj_center = view_points(org_center, view, normalize=normalize)[:2, 0]
            self.draw_text(canvas, proj_center[0], proj_center[1], str(self.label))

        if self.future_centers is not None:
            for mode_idx in range(self.num_modes):
                mode_prob = self.mode_probs[mode_idx]
                if mode_prob < self.RENDER_MODE_PROB_THRESHOLD:
                    # Do not render the low probability modes.
                    continue
                prev_x, prev_y, _ = self.center
                for horizon_idx in range(self.num_future_timesteps):
                    if self.num_future_timesteps > 1:
                        color_int = tuple(int(c * 255) for c in colors[0])
                        color = self.fade_color(color_int, horizon_idx, self.num_future_timesteps - 1)
                        color = tuple(c / 255 for c in color)
                    else:
                        color = colors[0]
                    waypoint = self.future_centers[mode_idx, horizon_idx]
                    if waypoint is not None and not np.isnan(waypoint).any():
                        next_x, next_y, _ = waypoint
                        alpha = max(1.0 - horizon_idx * 0.1, 0.1) * mode_prob
                        self.draw_line(
                            from_x=prev_x,
                            to_x=next_x,
                            from_y=prev_y,
                            to_y=next_y,
                            color=color,
                            marker=marker,
                            linewidth=linewidth,
                            canvas=canvas,
                            alpha=alpha,
                        )
                        prev_x, prev_y = next_x, next_y

    @staticmethod
    def fade_color(color: Tuple[int, int, int], step: int, total_number_of_steps: int) -> Tuple[int, int, int]:
        """
        Fades a color so that future observations are darker in the image.
        :param color: Tuple of ints describing an RGB color.
        :param step: The current time step.
        :param total_number_of_steps: The total number of time steps the agent has in the image.
        :return: Tuple representing faded rgb color.
        """
        LOWEST_VALUE = 0.2

        hsv_color = colorsys.rgb_to_hsv(*color)

        increment = (float(hsv_color[2]) / 255.0 - LOWEST_VALUE) / total_number_of_steps

        new_value = float(hsv_color[2]) / 255.0 - step * increment

        new_rgb = colorsys.hsv_to_rgb(float(hsv_color[0]), float(hsv_color[1]), new_value * 255.0)
        new_rgb_int = tuple(int(c) for c in new_rgb)

        return new_rgb_int  # type: ignore

    @staticmethod
    @functools.lru_cache()
    def _calc_corners(
        width: float, length: float, height: float, center: Tuple[float], rotation_matrix: Tuple[float]
    ) -> npt.NDArray[np.float64]:
        """
        Cached helper function to calculate corners from center and size.
        :param w: Width of box.
        :param l: Length of box.
        :param h: Height of box.
        :param center: Center of box.
        :param rotation_matrix: Rotation matrix of box.
        :return: Corners of box given as <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        corners = np.array(
            [[1, 1, 1, 1, -1, -1, -1, -1], [1, -1, -1, 1, 1, -1, -1, 1], [1, 1, -1, -1, 1, 1, -1, -1]], dtype=float
        )  # type: ignore
        corners[0] *= length / 2
        corners[1] *= width / 2
        corners[2] *= height / 2
        rot_mat = np.array(rotation_matrix).reshape(3, 3)  # type: ignore
        # Rotate
        corners = np.dot(rot_mat, corners)

        # Translate
        corners += np.array(center).reshape((-1, 1))

        return corners
