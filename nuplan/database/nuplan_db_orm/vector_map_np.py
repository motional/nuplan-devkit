from __future__ import annotations  # postpone evaluation of annotations

from typing import Dict, NamedTuple

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion


class VectorMapNp(NamedTuple):
    """
    Vector map data structure, including:
        coords: <np.float: num_lane_segments, 2, 2>
            The (x, y) coordinates of the start and end point of the lane segments.
        multi_scale_connections: Dict of {scale: connections_of_scale}.
            Each connections_of_scale is represented by an array of <np.float: num_connections, 2>,
            and each column in the array is [from_lane_segment_idx, to_lane_segment_idx].
    """

    coords: npt.NDArray[np.float64]
    multi_scale_connections: Dict[int, npt.NDArray[np.float64]]

    def translate(self, translate: npt.NDArray[np.float64]) -> VectorMapNp:
        """
        Translate the vector map.

        :param translate: <np.float: 3,>. Translation in x, y, z.
        :return: Translated vector map.
        """
        coords = self.coords
        coords += translate[:2]
        return self._replace(coords=coords)

    def rotate(self, quaternion: Quaternion) -> VectorMapNp:
        """
        Rotate the vector map.

        :param quaternion: Rotation to apply.
        :return: Rotated vector map.
        """
        coords = self.coords
        # Flattern the first two dimensions to make the shape (num_lane_segments * 2, 2).
        num_lane_segments, _, _ = coords.shape
        coords = coords.reshape(num_lane_segments * 2, 2)
        # Add zeros to the z dimension to make them 3D points.
        coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
        # Rotate.
        coords = np.dot(quaternion.rotation_matrix.astype(coords.dtype), coords)
        # Remove the z dimension and reshape it back to (num_lane_segments, 2, 2).
        coords = coords[:, :2].reshape(num_lane_segments, 2, 2)
        return self._replace(coords=coords)

    def scale(self, scale: npt.NDArray[np.float64]) -> VectorMapNp:
        """
        Scale the vector map.

        :param scale: <np.float: 3,>. Scale in x, y, z.
        :return: Scaled vector map.
        """
        # Ignore the z dimension.
        coords = self.coords
        coords *= scale[:2]
        return self._replace(coords=coords)

    def xflip(self) -> VectorMapNp:
        """
        Flip the vector map along the X-axis.
        :return: Flipped vector map.
        """
        coords = self.coords
        coords[:, :, 0] *= -1
        return self._replace(coords=coords)

    def yflip(self) -> VectorMapNp:
        """
        Flip the vector map along the Y-axis.
        :return: Flipped vector map.
        """
        coords = self.coords
        coords[:, :, 1] *= -1
        return self._replace(coords=coords)
