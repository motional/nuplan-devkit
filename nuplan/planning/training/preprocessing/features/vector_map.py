from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import torch
from nuplan.planning.script.builders.utils.utils_type import are_the_same_type, validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, \
    FeatureDataType, to_tensor
from pyquaternion import Quaternion


@dataclass
class VectorMap(AbstractModelFeature):
    """
    Vector map data struture, including:
        coords: List[<np.ndarray: num_lane_segments, 2, 2>].
            The (x, y) coordinates of the start and end point of the lane segments.
        multi_scale_connections: List[Dict of {scale: connections_of_scale}].
            Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
            and each column in the array is [from_lane_segment_idx, to_lane_segment_idx].

    In both cases, the List represent number of batches. This is a special feature where each batch entry
    can have different size. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """
    coords: List[FeatureDataType]
    multi_scale_connections: List[Dict[int, FeatureDataType]]
    _lane_coord_dim = 2

    def __post_init__(self) -> None:
        if len(self.coords) != len(self.multi_scale_connections):
            raise RuntimeError(
                f"Not consistent length of batches! {len(self.coords)} != {len(self.multi_scale_connections)}")

        if len(self.coords) == 0:
            raise RuntimeError("Batch size has to be > 0!")

        for coords in self.coords:
            if coords.shape[1] != 2 or coords.shape[2] != 2:
                raise RuntimeError("The dimension of coords is not correct!")

    @property
    def num_of_batches(self) -> int:
        """
        :return: number of batches
        """
        return len(self.coords)

    @classmethod
    def lane_coord_dim(cls) -> int:
        """
        :return: dimension of coords, should be 2 (x, y)
        """
        return cls._lane_coord_dim

    @classmethod
    def flatten_lane_coord_dim(cls) -> int:
        """
        :return: dimension of flattened start and end coords, should be 4 = 2 x (x, y)
        """
        return 2 * cls._lane_coord_dim

    @classmethod
    def collate(cls, batch: List[VectorMap]) -> VectorMap:
        return VectorMap(coords=[data for b in batch for data in b.coords],
                         multi_scale_connections=[data for b in batch for data in b.multi_scale_connections])

    def to_feature_tensor(self) -> VectorMap:
        """ Implemented. See interface. """
        return VectorMap(coords=[to_tensor(coords).contiguous() for coords in self.coords],
                         multi_scale_connections=[{scale: to_tensor(connection).contiguous()
                                                   for scale, connection in multi_scale_connections.items()} for
                                                  multi_scale_connections in self.multi_scale_connections])

    def to_device(self, device: torch.device) -> VectorMap:
        """ Implemented. See interface. """
        return VectorMap(coords=[coords.to(device=device) for coords in self.coords],
                         multi_scale_connections=[{scale: connection.to(device=device)
                                                   for scale, connection in multi_scale_connections.items()} for
                                                  multi_scale_connections in self.multi_scale_connections])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VectorMap:
        """ Implemented. See interface. """
        return VectorMap(coords=data["coords"],
                         multi_scale_connections=data["multi_scale_connections"])

    def rotate(self, quaternion: Quaternion) -> VectorMap:
        """
        Rotate the vector map.

        :param quaternion: Rotation to apply.
        """
        if not all(validate_type(coord, np.ndarray) for coord in self.coords):
            raise RuntimeError("This function works only with numpy arrays!")

        def rotate_coord(coords: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # Flatten the first two dimensions to make the shape (num_lane_segments * 2, 2).
            num_lane_segments, _, _ = coords.shape
            coords = coords.reshape(num_lane_segments * 2, 2)

            # Add zeros to the z dimension to make them 3D points.
            coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)  # type: ignore

            # Rotate.
            coords = np.dot(quaternion.rotation_matrix.astype(coords.dtype), coords)  # type: ignore

            # Remove the z dimension and reshape it back to (num_lane_segments, 2, 2).
            return coords[:, :2].reshape(num_lane_segments, 2, 2)  # type: ignore

        return VectorMap(coords=[rotate_coord(data) for data in self.coords],
                         multi_scale_connections=self.multi_scale_connections)

    def translate(self, translation_value: FeatureDataType) -> VectorMap:
        """
        Translate the vector map.
        :param translation_value: Translation in x, y, z.
        """
        assert translation_value.ndim == 3, "Translation value must have dimension of 3 (x, y, z)"
        are_the_same_type(translation_value, self.coords[0])

        def translate_coord(coords: FeatureDataType) -> FeatureDataType:
            return coords + translation_value[:2]

        return VectorMap(coords=[translate_coord(coords) for coords in self.coords],
                         multi_scale_connections=self.multi_scale_connections)

    def scale(self, scale_value: FeatureDataType) -> VectorMap:
        """
        Scale the vector map.
        :param scale_value: <np.float: 3,>. Scale in x, y, z.
        """
        assert scale_value.ndim == 3, f"Scale value has incorrect dimension: {scale_value.ndim}!"
        are_the_same_type(scale_value, self.coords[0])

        def scale_coord(coords: FeatureDataType) -> FeatureDataType:
            return coords * scale_value[:2]

        return VectorMap(coords=[scale_coord(coords) for coords in self.coords],
                         multi_scale_connections=self.multi_scale_connections)

    def xflip(self) -> VectorMap:
        """
        Flip the vector map along the X-axis.
        """

        def xflip_coord(coords: FeatureDataType) -> FeatureDataType:
            return coords[:, :, 0] * -1

        return VectorMap(coords=[xflip_coord(coords) for coords in self.coords],
                         multi_scale_connections=self.multi_scale_connections)

    def yflip(self) -> VectorMap:
        """
        Flip the vector map along the Y-axis.
        """

        def yflip_coord(coords: FeatureDataType) -> FeatureDataType:
            return coords[:, :, 1] * -1

        return VectorMap(coords=[yflip_coord(coords) for coords in self.coords],
                         multi_scale_connections=self.multi_scale_connections)
