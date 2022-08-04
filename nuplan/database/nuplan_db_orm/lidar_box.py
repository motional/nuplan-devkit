from __future__ import annotations  # postpone evaluation of annotations

import bisect
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from pyquaternion import Quaternion
from sqlalchemy import Column, inspect
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey
from sqlalchemy.types import Float, Integer

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.database.common import sql_types
from nuplan.database.common.utils import simple_repr
from nuplan.database.nuplan_db_orm.category import Category
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.models import Base
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.iterable_lidar_box import IterableLidarBox
from nuplan.database.utils.label.utils import local2agent_type, raw_mapping
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory

if TYPE_CHECKING:
    from nuplan.database.nuplan_db_orm.log import Log

LIDAR_BOX_LRU_CACHE_SIZE = 2048

logger = logging.getLogger()


class LidarBox(Base):
    """
    Lidar box from tracker.
    """

    __tablename__ = "lidar_box"

    token: str = Column(sql_types.HexLen8, primary_key=True)
    lidar_pc_token: str = Column(sql_types.HexLen8, ForeignKey("lidar_pc.token"), nullable=False)
    track_token: str = Column(sql_types.HexLen8, ForeignKey("track.token"))
    next_token = Column(sql_types.HexLen8, ForeignKey("lidar_box.token"), nullable=True)  # type: str
    prev_token = Column(sql_types.HexLen8, ForeignKey("lidar_box.token"), nullable=True)  # type: str
    x: float = Column(Float)
    y: float = Column(Float)
    z: float = Column(Float)
    width: float = Column(Float)
    length: float = Column(Float)
    height: float = Column(Float)
    vx: float = Column(Float)
    vy: float = Column(Float)
    vz: float = Column(Float)
    yaw: float = Column(Float)
    confidence: float = Column(Float)

    next = relationship("LidarBox", foreign_keys=[next_token], remote_side=[token])  # type: LidarBox
    prev = relationship("LidarBox", foreign_keys=[prev_token], remote_side=[token])  # type: LidarBox

    @property
    def _session(self) -> Any:
        """
        Get the underlying session.
        :return: The underlying session.
        """
        return inspect(self).session

    def __iter__(self) -> IterableLidarBox:
        """
        Returns a iterator object for LidarBox.
        :return: The iterator object.
        """
        return IterableLidarBox(self)

    def __reversed__(self) -> IterableLidarBox:
        """
        Returns a iterator object for LidarBox that traverses in reverse.
        :return: The iterator object.
        """
        return IterableLidarBox(self, reverse=True)

    def __repr__(self) -> str:
        """
        Return the string representation.
        :return: The string representation.
        """
        desc: str = simple_repr(self)
        return desc

    @property
    def log(self) -> Log:
        """
        Returns the Log containing the LidarBox.
        :return: The log containing the lidar box.
        """
        return self.lidar_pc.log

    @property
    def category(self) -> Category:
        """
        Returns the Category of the LidarBox.
        :return: The category of the lidar box.
        """
        return self.track.category

    @property
    def timestamp(self) -> int:
        """
        Returns the timestamp of the LidarBox.
        :return: The timestamp of the lidar box.
        """
        return int(self.lidar_pc.timestamp)

    @property
    def distance_to_ego(self) -> float:
        """
        Returns the distance of detection from Ego Vehicle.
        :return: The distance to ego vehicle.
        """
        return float(np.sqrt((self.x - self.lidar_pc.ego_pose.x) ** 2 + (self.y - self.lidar_pc.ego_pose.y) ** 2))

    @property
    def size(self) -> List[float]:
        """
        Get the box size.
        :return: The box size.
        """
        return [self.width, self.length, self.height]

    @property
    def translation(self) -> List[float]:
        """
        Get the box location.
        :return: The box location.
        """
        return [self.x, self.y, self.z]

    @property
    def rotation(self) -> List[float]:
        """
        Get the box rotation in euler angles.
        :return: The box rotation in euler angles.
        """
        qx = Quaternion(axis=(1, 0, 0), radians=0.0)
        qy = Quaternion(axis=(0, 1, 0), radians=0.0)
        qz = Quaternion(axis=(0, 0, 1), radians=self.yaw)

        return list(qx * qy * qz)

    @property
    def quaternion(self) -> Quaternion:
        """
        Get the box rotation in quaternion.
        :return: The box rotation in quaternion.
        """
        return Quaternion(self.rotation)

    @property
    def translation_np(self) -> npt.NDArray[np.float64]:
        """
        Get the box translation in numpy.
        :return: <np.float: 3> Translation.
        """
        return np.array(self.translation)

    @property
    def size_np(self) -> npt.NDArray[np.float64]:
        """
        Get the box size in numpy.
        :return: <np.float, 3> Width, length and height.
        """
        return np.array(self.size)

    @cached(
        cache=LRUCache(maxsize=LIDAR_BOX_LRU_CACHE_SIZE), key=lambda self: hashkey(self.track_token)
    )  # type: ignore
    def _get_box_items(self) -> Tuple[List[Integer], List[LidarBox]]:
        """
        Get all boxes along the track.
        :return: The list of timestamps and boxes along the track.
        """
        box_list: List[LidarBox] = self._session.query(LidarBox).filter(LidarBox.track_token == self.track_token).all()
        sorted_box_list = sorted(box_list, key=lambda x: x.timestamp)

        return [b.timestamp for b in sorted_box_list], sorted_box_list

    @cached(
        cache=LRUCache(maxsize=LIDAR_BOX_LRU_CACHE_SIZE), key=lambda self: hashkey(self.track_token)
    )  # type: ignore
    def get_box_items_to_iterate(self) -> Dict[int, Tuple[Optional[LidarBox], Optional[LidarBox]]]:
        """
        Get all boxes along the track.
        :return: Dict. Key is timestamp of box, value is Tuple of (prev,next) LidarBox.
        """
        box_list = self._session.query(LidarBox).filter(LidarBox.track_token == self.track_token).all()
        sorted_box_list = sorted(box_list, key=lambda x: x.timestamp)  # type: ignore

        return {
            box.timestamp: (prev, next)
            for box, prev, next in zip(sorted_box_list, [None] + sorted_box_list[:-1], sorted_box_list[1:] + [None])
        }

    def _find_box(self, step: int = 0) -> Optional[LidarBox]:
        """
        Find the next box along the track with the given step.
        :param: step: The number of steps to look ahead, defaults to zero.
        :return: The found box if any.
        """
        timestamp_list, sorted_box_list = self._get_box_items()
        i = bisect.bisect_left(timestamp_list, self.timestamp)
        j = i + step
        if j < 0 or j >= len(sorted_box_list):
            return None

        return sorted_box_list[j]

    def future_or_past_ego_poses(self, number: int, mode: str, direction: str) -> List[EgoPose]:
        """
        Get n future or past vehicle poses. Note here the frequency of pose differs from frequency of LidarBox.
        :param number: Number of poses to fetch or number of seconds of ego poses to fetch.
        :param mode: Either n_poses or n_seconds.
        :param direction: Future or past ego poses to fetch, could be 'prev' or 'next'.
        :return: List of up to n or n seconds future or past ego poses.
        """
        if direction == 'prev':
            if mode == 'n_poses':
                return (  # type: ignore
                    self._session.query(EgoPose)
                    .filter(
                        EgoPose.timestamp < self.lidar_pc.ego_pose.timestamp,
                        self.lidar_pc.lidar.log_token == EgoPose.log_token,
                    )
                    .order_by(EgoPose.timestamp.desc())
                    .limit(number)
                    .all()
                )
            elif mode == 'n_seconds':
                return (  # type: ignore
                    self._session.query(EgoPose)
                    .filter(
                        EgoPose.timestamp - self.lidar_pc.ego_pose.timestamp < 0,
                        EgoPose.timestamp - self.lidar_pc.ego_pose.timestamp >= -number * 1e6,
                        self.lidar_pc.lidar.log_token == EgoPose.log_token,
                    )
                    .order_by(EgoPose.timestamp.desc())
                    .all()
                )
            else:
                raise ValueError(f"Unknown mode: {mode}.")
        elif direction == 'next':
            if mode == 'n_poses':
                return (  # type: ignore
                    self._session.query(EgoPose)
                    .filter(
                        EgoPose.timestamp > self.lidar_pc.ego_pose.timestamp,
                        self.lidar_pc.lidar.log_token == EgoPose.log_token,
                    )
                    .order_by(EgoPose.timestamp.asc())
                    .limit(number)
                    .all()
                )
            elif mode == 'n_seconds':
                return (  # type: ignore
                    self._session.query(EgoPose)
                    .filter(
                        EgoPose.timestamp - self.lidar_pc.ego_pose.timestamp > 0,
                        EgoPose.timestamp - self.lidar_pc.ego_pose.timestamp <= number * 1e6,
                        self.lidar_pc.lidar.log_token == EgoPose.log_token,
                    )
                    .order_by(EgoPose.timestamp.asc())
                    .all()
                )
            else:
                raise ValueError(f"Unknown mode: {mode}.")
        else:
            raise ValueError(f"Unknown direction: {direction}.")

    def _temporal_neighbors(self) -> Tuple[LidarBox, LidarBox, bool, bool]:
        """
        Find temporal neighbors to calculate velocity and angular velocity.
        :return: The previous box, next box and their existences. If the previous or next box do not exist, they will
            be set to the current box itself.
        """
        has_prev = self.prev is not None
        has_next = self.next is not None

        if has_prev:
            prev_lidar_box = self.prev
        else:
            prev_lidar_box = self

        if has_next:
            next_lidar_box = self.next
        else:
            next_lidar_box = self

        return prev_lidar_box, next_lidar_box, has_prev, has_next

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        """
        Estimate box velocity for a box.
        :return: The estimated box velocity of the box.
        """
        max_time_diff = 1.5
        prev_lidar_box, next_lidar_box, has_prev, has_next = self._temporal_neighbors()

        if not has_prev and not has_next:
            # Can't estimate velocity for a single annotation
            return np.array([np.nan, np.nan, np.nan])

        pos_next: npt.NDArray[np.float64] = np.array(next_lidar_box.translation)
        pos_prev: npt.NDArray[np.float64] = np.array(prev_lidar_box.translation)
        pos_diff: npt.NDArray[np.float64] = pos_next - pos_prev
        pos_diff[2] = 0  # We don't have robust localization in z. So set this to zero.

        time_next = 1e-6 * next_lidar_box.timestamp
        time_prev = 1e-6 * prev_lidar_box.timestamp
        time_diff = time_next - time_prev

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    @property
    def angular_velocity(self) -> float:
        """
        Estimate box angular velocity for a box.
        :return: The estimated box angular velocity of the box.
        """
        max_time_diff = 1.5
        prev_lidar_box, next_lidar_box, has_prev, has_next = self._temporal_neighbors()

        if not has_prev and not has_next:
            # Can't estimate angular velocity for a single annotation
            return np.nan

        time_next = 1e-6 * next_lidar_box.timestamp
        time_prev = 1e-6 * prev_lidar_box.timestamp
        time_diff = time_next - time_prev

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.nan
        else:
            # We currently only look at yaw
            yaw_diff = next_lidar_box.yaw - prev_lidar_box.yaw

            # Yaw in radians, in the range `[-pi, pi]`. Hence, raw yaw_diff is in tha range `[-2pi, 2pi]`
            # Assume all actors heading changes are small within `max_time_diff`, compensate the changes to [-pi, pi]
            if yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            elif yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi
            return float(yaw_diff / time_diff)

    def box(self) -> Box3D:
        """
        Get the Box3D representation of the box.
        :return: The box3d representation of the box.
        """
        label_local = raw_mapping['global2local'][self.category.name]
        label_int = raw_mapping['local2id'][label_local]
        return Box3D(
            center=self.translation,
            size=self.size,
            orientation=self.quaternion,
            token=self.token,
            label=label_int,
            track_token=self.track_token,
        )

    def tracked_object(self, future_waypoints: Optional[List[Waypoint]]) -> TrackedObject:
        """
        Creates an Agent object
        :param future_waypoints: Optional future poses, which will be used as predicted trajectory
        """
        pose = StateSE2(self.translation[0], self.translation[1], self.yaw)

        oriented_box = OrientedBox(pose, width=self.size[0], length=self.size[1], height=self.size[2])

        label_local = raw_mapping['global2local'][self.category.name]
        tracked_object_type = TrackedObjectType[local2agent_type[label_local]]

        if tracked_object_type in AGENT_TYPES:
            return Agent(
                tracked_object_type=tracked_object_type,
                oriented_box=oriented_box,
                velocity=StateVector2D(self.vx, self.vy),
                predictions=[PredictedTrajectory(1.0, future_waypoints)]  # Probability of 1 as is from future waypoints
                if future_waypoints
                else [],
                angular_velocity=np.nan,
                metadata=SceneObjectMetadata(
                    token=self.token,
                    track_token=self.track_token,
                    track_id=None,
                    timestamp_us=self.timestamp,
                    category_name=self.category.name,
                ),
            )
        else:
            return StaticObject(
                tracked_object_type=tracked_object_type,
                oriented_box=oriented_box,
                metadata=SceneObjectMetadata(
                    token=self.token,
                    track_token=self.track_token,
                    track_id=None,
                    timestamp_us=self.timestamp,
                    category_name=self.category.name,
                ),
            )
