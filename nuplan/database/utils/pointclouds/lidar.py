from __future__ import annotations

from typing import IO, Any, ByteString, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pytest
from matplotlib import axes, cm
from PIL import Image
from pyquaternion import Quaternion

from nuplan.database.utils.geometry import view_points
from nuplan.database.utils.plot import rainbow
from nuplan.database.utils.pointclouds.pointcloud import PointCloud

# Field name used to find lidar sweep time data in pcd files.
PCD_TIMESTAMP_FIELD_NAME = 'time_delta'


def pcd_to_numpy(pcd_file: str) -> npt.NDArray[np.float32]:
    """
    This function converts the pointcloud *.pcl or *.pcd files to numpy (x, y, z, i) format,
    or (x, y, z, i, t) format if a time field is present.
    :param pcd_file: Name of the point cloud file (*.pcl or *.pcd)
    :return: A numpy array of shape (n, 4) or (n, 5), dtype = np.float32
    """
    with open(pcd_file) as ifile:
        data = [line.strip() for line in ifile]

    meta = data[:10]
    assert meta[0].startswith('#'), 'First line must be comment'
    assert meta[1].startswith('VERSION'), 'Second line must be VERSION'

    # FIELDS is the third line in the file. Remove the word FIELDS so the first field name is at index 0 in this list.
    fields = meta[2].split(' ')[1:]
    assert all(f in fields for f in ['x', 'y', 'z']), 'x, y, and z fields are required'

    # not support binary.
    assert data[10] == 'DATA ascii'

    data = data[11:]  # remove header stuff
    data = [d.split(' ') for d in data]  # type: ignore  # split each line

    # Currently we ignore the SIZE and TYPE lines in pcd files and assume everything's a 32-bit float.
    all_columns = np.array(data, dtype=np.float32)  # type: ignore
    num_points = all_columns.shape[0]
    has_delta_time = PCD_TIMESTAMP_FIELD_NAME in fields
    result_shape = (num_points, 5) if has_delta_time else (num_points, 4)
    result = np.zeros(result_shape, dtype=np.float32)  # type: ignore

    result[:, 0] = all_columns[:, fields.index('x')]
    result[:, 1] = all_columns[:, fields.index('y')]
    result[:, 2] = all_columns[:, fields.index('z')]
    if 'intensity' in fields:
        result[:, 3] = all_columns[:, fields.index('intensity')]
    if has_delta_time:
        result[:, 4] = all_columns[:, fields.index(PCD_TIMESTAMP_FIELD_NAME)]

    return result


class LidarPointCloud:
    """Simple data class representing a point cloud."""

    def __init__(self, points: npt.NDArray[np.float32]) -> None:
        """
        Class for manipulating and viewing point clouds.
        :param points: <np.float: f, n>. Input point cloud matrix with f features per point and n points.
        """
        if points.ndim == 1:
            points = np.atleast_2d(points).T

        self.points = points

    @staticmethod
    def load_pcd_bin(pcd_bin: Union[str, IO[Any], ByteString], pcd_bin_version: int = 1) -> npt.NDArray[np.float32]:
        """
        Loads from pcd binary format:
            version 1: a numpy array with 5 cols (x, y, z, intensity, ring).
            version 2: a numpy array with 6 cols (x, y, z, intensity, ring, lidar_id).
        :param pcd_bin: File path or a file-like object or raw bytes.
        :param pcd_bin_version: 1 or 2, see above.
        :return: <np.float: 6, n>. Point cloud matrix[(x, y, z, intensity, ring, lidar_id)].
        """
        if isinstance(pcd_bin, str):
            scan = np.fromfile(pcd_bin, dtype=np.float32)  # type: ignore
        else:
            if not isinstance(pcd_bin, bytes):
                pcd_bin = pcd_bin.read()  # type: ignore
            scan = np.frombuffer(pcd_bin, dtype=np.float32)  # type: ignore
            # frombuffer returns a read-only np.array
            scan = np.copy(scan)

        if pcd_bin_version == 1:
            points = scan.reshape((-1, 5))
            # Append lidar_id column
            points = np.hstack((points, -1 * np.ones((points.shape[0], 1), dtype=np.float32)))
        elif pcd_bin_version == 2:
            points = scan.reshape((-1, 6))
        else:
            pytest.fail('Unknown pcd bin file version: %d' % pcd_bin_version)

        return points.T

    @staticmethod
    def load_pcd(pcd_data: Union[IO[Any], ByteString]) -> npt.NDArray[np.float32]:
        """
        Loads a pcd file.
        :param pcd_data: File path or a file-like object or raw bytes.
        :return: <np.float: 6, n>. Point cloud matrix[(x, y, z, intensity, ring, lidar_id)].
        """
        if not isinstance(pcd_data, bytes):
            pcd_data = pcd_data.read()  # type: ignore

        return PointCloud.parse(pcd_data).to_pcd_bin2()  # type: ignore

    @classmethod
    def from_file(cls, file_name: str) -> LidarPointCloud:
        """
        Instantiates from a .pcl, .pcd, .npy, or .bin file.
        :param file_name: Path of the pointcloud file on disk.
        :return: A LidarPointCloud object.
        """
        if file_name.endswith('.bin'):
            points = cls.load_pcd_bin(file_name, 1)
        elif file_name.endswith('.bin2'):
            points = cls.load_pcd_bin(file_name, 2)
        elif file_name.endswith('.pcl') or file_name.endswith('.pcd'):
            points = pcd_to_numpy(file_name).T
        elif file_name.endswith('.npy'):
            points = np.load(file_name)
        else:
            raise ValueError('Unsupported filetype {}'.format(file_name))

        return cls(points)

    @classmethod
    def from_buffer(cls, pcd_data: Union[IO[Any], ByteString], content_type: str = 'bin') -> LidarPointCloud:
        """
        Instantiates from buffer.
        :param pcd_data: File path or a file-like object or raw bytes.
        :param content_type: Type of the point cloud content, such as 'bin', 'bin2', 'pcd'.
        :return: A LidarPointCloud object.
        """
        if content_type == 'bin':
            return cls(cls.load_pcd_bin(pcd_data, 1))
        elif content_type == 'bin2':
            return cls(cls.load_pcd_bin(pcd_data, 2))
        elif content_type == 'pcd':
            return cls(cls.load_pcd(pcd_data))
        else:
            raise NotImplementedError('Not implemented content type: %s' % content_type)

    @classmethod
    def make_random(cls) -> LidarPointCloud:
        """
        Instantiates a random point cloud.
        :return: LidarPointCloud instance.
        """
        return LidarPointCloud(points=np.random.normal(0, 100, size=(4, 100)))  # type: ignore

    def __eq__(self, other: object) -> bool:
        """
        Checks if two LidarPointCloud are equal.
        :param other: Other object.
        :return: True if both objects are equal otherwise False.
        """
        if not isinstance(other, LidarPointCloud):
            return NotImplemented

        return np.allclose(self.points, other.points, atol=1e-06)

    def copy(self) -> LidarPointCloud:
        """
        Creates a copy of self.
        :return: LidarPointCloud instance.
        """
        return LidarPointCloud(points=self.points.copy())

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return int(self.points.shape[1])

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        assert 0 < ratio < 1

        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, min_dist: float) -> None:
        """
        Removes points too close within a certain distance from origin from bird view (so dist = sqrt(x^2+y^2)).
        :param min_dist: The distance threshold.
        """
        dist_from_orig = np.linalg.norm(self.points[:2, :], axis=0)
        self.points = self.points[:, dist_from_orig >= min_dist]

    def radius_filter(self, radius: float) -> None:
        """
        Removes points outside the given radius.
        :param radius: Radius in meters.
        """
        keep = np.sqrt(self.points[0] ** 2 + self.points[1] ** 2) <= radius
        self.points = self.points[:, keep]

    def range_filter(
        self,
        xrange: Tuple[float, float] = (-np.inf, np.inf),
        yrange: Tuple[float, float] = (-np.inf, np.inf),
        zrange: Tuple[float, float] = (-np.inf, np.inf),
    ) -> None:
        """
        Restricts points to specified ranges.
        :param xrange: (xmin, xmax).
        :param yrange: (ymin, ymax).
        :param zrange: (zmin, zmax).
        """
        # Figure out which points to keep.
        keep_x = np.logical_and(xrange[0] <= self.points[0], self.points[0] <= xrange[1])
        keep_y = np.logical_and(yrange[0] <= self.points[1], self.points[1] <= yrange[1])
        keep_z = np.logical_and(zrange[0] <= self.points[2], self.points[2] <= zrange[1])
        keep = np.logical_and(keep_x, np.logical_and(keep_y, keep_z))

        self.points = self.points[:, keep]

    def translate(self, x: npt.NDArray[np.float64]) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3,>. Translation in x, y, z.
        """
        self.points[:3] += x.reshape((-1, 1))

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Applies a rotation.
        :param quaternion: Rotation to apply.
        """
        self.points[:3] = np.dot(quaternion.rotation_matrix.astype(np.float32), self.points[:3])

    def transform(self, transf_matrix: npt.NDArray[np.float64]) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogeneous transformation matrix.
        """
        transf_matrix = transf_matrix.astype(np.float32)
        self.points[:3, :] = transf_matrix[:3, :3] @ self.points[:3] + transf_matrix[:3, 3].reshape((-1, 1))

    def scale(self, scale: Tuple[float, float, float]) -> None:
        """
        Scales the lidar xyz coordinates.
        :param scale: The scaling parameter.
        """
        scale_arr = np.array(scale)  # type: ignore
        scale_arr.shape = (3, 1)  # Make sure it is a column vector.
        self.points[:3, :] *= np.tile(scale_arr, (1, self.nbr_points()))

    def render_image(
        self,
        canvas_size: Tuple[int, int] = (1001, 1001),
        view: npt.NDArray[np.float64] = np.array([[10, 0, 0, 500], [0, 10, 0, 500], [0, 0, 10, 0]]),
        color_dim: int = 2,
    ) -> Image.Image:
        """
        Renders pointcloud to an array with 3 channels appropriate for viewing as an image. The image is color coded
        according the color_dim dimension of points (typically the height).
        :param canvas_size: (width, height). Size of the canvas on which to render the image.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param color_dim: The dimension of the points to be visualized as color. Default is 2 for height.
        :return: A Image instance.
        """
        # Apply desired transformation to the point cloud. (height is here considered independent of the view).
        heights = self.points[2, :]
        points = view_points(self.points[:3, :], view, normalize=False)
        points[2, :] = heights

        # Remove points that fall outside the canvas.
        mask = np.ones(points.shape[1], dtype=bool)  # type: ignore
        mask = np.logical_and(mask, points[0, :] < canvas_size[0] - 1)
        mask = np.logical_and(mask, points[0, :] > 0)
        mask = np.logical_and(mask, points[1, :] < canvas_size[1] - 1)
        mask = np.logical_and(mask, points[1, :] > 0)
        points = points[:, mask]

        # Scale color_values to be between 0 and 255.
        color_values = points[color_dim, :]
        color_values = 255.0 * (color_values - np.amin(color_values)) / (np.amax(color_values) - np.amin(color_values))

        # Rounds to ints and generate colors that will be used in the image.
        points = np.int16(np.round(points[:2, :]))
        color_values = np.int16(np.round(color_values))
        cmap = [cm.jet(i / 255, bytes=True)[:3] for i in range(256)]

        # Populate canvas, use maximum color_value for each bin
        render = np.tile(np.expand_dims(np.zeros(canvas_size, dtype=np.uint8), axis=2), [1, 1, 3])  # type: ignore
        color_value_array: npt.NDArray[np.float64] = -1 * np.ones(canvas_size, dtype=float)  # type: ignore
        for (col, row), color_value in zip(points.T, color_values.T):
            if color_value > color_value_array[row, col]:
                color_value_array[row, col] = color_value
                render[row, col] = cmap[color_value]

        return Image.fromarray(render)

    def render_height(
        self,
        ax: axes.Axes,
        view: npt.NDArray[np.float64] = np.eye(4),
        x_lim: Tuple[float, float] = (-20, 20),
        y_lim: Tuple[float, float] = (-20, 20),
        marker_size: float = 1,
    ) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(self.points[2, :], ax, view, x_lim, y_lim, marker_size)

    def render_intensity(
        self,
        ax: axes.Axes,
        view: npt.NDArray[np.float64] = np.eye(4),
        x_lim: Tuple[float, float] = (-20, 20),
        y_lim: Tuple[float, float] = (-20, 20),
        marker_size: float = 1,
    ) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(self.points[3, :], ax, view, x_lim, y_lim, marker_size)

    def render_label(
        self,
        ax: axes.Axes,
        id2color: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        view: npt.NDArray[np.float64] = np.eye(4),
        x_lim: Tuple[float, float] = (-20, 20),
        y_lim: Tuple[float, float] = (-20, 20),
        marker_size: float = 1.0,
    ) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points. Each points is colored based
        on labels through the label color mapping, If no mapping provided, we use the rainbow function to assign
        the colors.
        :param id2color: {label_id : (R, G, B, A)}. Id to color mapping where RGBA is within [0, 255].
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        # Label ids are expected to be concatenated at the end of the point feature.
        label = self.points[-1]
        colors: Dict[int, Tuple[Any, ...]] = {}
        if id2color is None:
            unique_label = np.unique(label)  # type: ignore
            color_rainbow = rainbow(len(unique_label), normalized=True)
            for label_id, c in zip(unique_label, color_rainbow):
                colors[label_id] = c
        else:
            for key, color in id2color.items():
                # The expected color info for matlab is normalized
                colors[key] = np.array(color) / 255.0  # type: ignore
        # Transparent if not in the id list.
        color_list = list(map(lambda x: colors.get(x, np.array((1.0, 1.0, 1.0, 0.0))), label))  # type: ignore

        self._render_helper(color_list, ax, view, x_lim, y_lim, marker_size)  # type: ignore

    def _render_helper(
        self,
        colors: Union[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]],
        ax: axes.Axes,
        view: npt.NDArray[np.float64],
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        marker_size: float,
    ) -> None:
        """
        Helper function for rendering.
        :param colors: Array-like or list of colors or color input for scatter function.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=colors, s=marker_size)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
