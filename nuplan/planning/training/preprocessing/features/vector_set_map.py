from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pyquaternion import Quaternion

from nuplan.planning.script.builders.utils.utils_type import are_the_same_type, validate_type
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneSegmentTrafficLightData,
    VectorFeatureLayer,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)
from nuplan.planning.training.preprocessing.features.vector_utils import (
    rotate_coords,
    scale_coords,
    translate_coords,
    xflip_coords,
    yflip_coords,
)


@dataclass
class VectorSetMap(AbstractModelFeature):
    """
    Vector set map data structure, including:
        coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample in batch,
                indexed by map feature.
        traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample
                in batch, indexed by map feature. Same indexing as coords.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
        availabilities: Dict[str, List[<np.ndarray: num_elements, num_points>]].
            Boolean indicator of whether feature data (coords as well as traffic light status if it exists for feature)
                is available for point at given index or if it is zero-padded.

    Feature formulation as sets of vectors for each map element similar to that of VectorNet ("VectorNet: Encoding HD
    Maps and Agent Dynamics from Vectorized Representation"), except map elements are encoded as sets of singular x, y
    points instead of start, end point pairs.

    Coords, traffic light status, and availabilities data are each keyed by map feature name, with dimensionality
    (availabilities don't include feature dimension):
    B: number of samples per batch (variable)
    N: number of map elements (fixed for a given map feature)
    P: number of points (fixed for a given map feature)
    F: number of features (2 for coords, 4 for traffic light status)

    Data at the same index represent the same map element/point among coords, traffic_light_data, and availabilities,
    with traffic_light_data only optionally included. For each map feature, the top level List represents number of
    samples per batch. This is a special feature where each batch entry can have a different size. For that reason, the
    features can not be placed to a single tensor, and we batch the feature with a custom `collate` function.
    """

    coords: Dict[str, List[FeatureDataType]]
    traffic_light_data: Dict[str, List[FeatureDataType]]
    availabilities: Dict[str, List[FeatureDataType]]
    _polyline_coord_dim: int = 2
    _traffic_light_status_dim: int = LaneSegmentTrafficLightData.encoding_dim()

    def __post_init__(self) -> None:
        """
        Sanitize attributes of the dataclass.
        :raise RuntimeError if dimensions invalid.
        """
        # Check empty data
        if not len(self.coords) > 0:
            raise RuntimeError("Coords cannot be empty!")

        if not all([len(coords) > 0 for coords in self.coords.values()]):
            raise RuntimeError("Batch size has to be > 0!")

        self._sanitize_feature_consistency()
        self._sanitize_data_dimensionality()

    def _sanitize_feature_consistency(self) -> None:
        """
        Check data dimensionality consistent across and within map features.
        :raise RuntimeError if dimensions invalid.
        """
        # Check consistency across map features
        if not all([len(coords) == len(list(self.coords.values())[0]) for coords in self.coords.values()]):
            raise RuntimeError("Batch size inconsistent across features!")

        # Check consistency across data within map feature
        for feature_name, feature_coords in self.coords.items():
            if feature_name not in self.availabilities:
                raise RuntimeError("No matching feature in coords for availabilities data!")
            feature_avails = self.availabilities[feature_name]

            if len(feature_avails) != len(feature_coords):
                raise RuntimeError(
                    f"Batch size between coords and availabilities data inconsistent! {len(feature_coords)} != {len(feature_avails)}"
                )
            feature_size = self.feature_size(feature_name)
            if feature_size[1] == 0:
                raise RuntimeError("Features cannot be empty!")

            for coords in feature_coords:
                if coords.shape[0:2] != feature_size:
                    raise RuntimeError(
                        f"Coords for {feature_name} feature don't have consistent feature size! {coords.shape[0:2] != feature_size}"
                    )
            for avails in feature_avails:
                if avails.shape[0:2] != feature_size:
                    raise RuntimeError(
                        f"Availabilities for {feature_name} feature don't have consistent feature size! {avails.shape[0:2] != feature_size}"
                    )

        for feature_name, feature_tl_data in self.traffic_light_data.items():
            if feature_name not in self.coords:
                raise RuntimeError("No matching feature in coords for traffic light data!")
            feature_coords = self.coords[feature_name]

            if len(feature_tl_data) != len(self.coords[feature_name]):
                raise RuntimeError(
                    f"Batch size between coords and traffic light data inconsistent! {len(feature_coords)} != {len(feature_tl_data)}"
                )
            feature_size = self.feature_size(feature_name)

            for tl_data in feature_tl_data:
                if tl_data.shape[0:2] != feature_size:
                    raise RuntimeError(
                        f"Traffic light data for {feature_name} feature don't have consistent feature size! {tl_data.shape[0:2] != feature_size}"
                    )

    def _sanitize_data_dimensionality(self) -> None:
        """
        Check data dimensionality as expected.
        :raise RuntimeError if dimensions invalid.
        """
        for feature_coords in self.coords.values():
            for sample in feature_coords:
                if sample.shape[2] != self._polyline_coord_dim:
                    raise RuntimeError("The dimension of coords is not correct!")

        for feature_tl_data in self.traffic_light_data.values():
            for sample in feature_tl_data:
                if sample.shape[2] != self._traffic_light_status_dim:
                    raise RuntimeError("The dimension of traffic light data is not correct!")

        for feature_avails in self.availabilities.values():
            for sample in feature_avails:
                if len(sample.shape) != 2:
                    raise RuntimeError("The dimension of availabilities is not correct!")

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return (
            all([len(feature_coords) > 0 for feature_coords in self.coords.values()])
            and all([feature_coords[0].shape[0] > 0 for feature_coords in self.coords.values()])
            and all([feature_coords[0].shape[1] > 0 for feature_coords in self.coords.values()])
            and all([len(feature_tl_data) > 0 for feature_tl_data in self.traffic_light_data.values()])
            and all([feature_tl_data[0].shape[0] > 0 for feature_tl_data in self.traffic_light_data.values()])
            and all([feature_tl_data[0].shape[1] > 0 for feature_tl_data in self.traffic_light_data.values()])
            and all([len(features_avails) > 0 for features_avails in self.availabilities.values()])
            and all([features_avails[0].shape[0] > 0 for features_avails in self.availabilities.values()])
            and all([features_avails[0].shape[1] > 0 for features_avails in self.availabilities.values()])
        )

    @property
    def batch_size(self) -> int:
        """
        Batch size across features.
        :return: number of batches.
        """
        return len(list(self.coords.values())[0])  # All features guaranteed to have same batch size

    def feature_size(self, feature_name: str) -> Tuple[int, int]:
        """
        Number of map elements for given feature, points per element.
        :param feature_name: name of map feature to access.
        :return: [num_elements, num_points]
        :raise: RuntimeError if empty feature.
        """
        map_feature = self.coords[feature_name][0]
        if map_feature.size == 0:
            raise RuntimeError("Feature is empty!")
        return map_feature.shape[0], map_feature.shape[1]

    @classmethod
    def coord_dim(cls) -> int:
        """
        Coords dimensionality, should be 2 (x, y).
        :return: dimension of coords.
        """
        return cls._polyline_coord_dim

    @classmethod
    def traffic_light_status_dim(cls) -> int:
        """
        Traffic light status dimensionality, should be 4.
        :return: dimension of traffic light status.
        """
        return cls._traffic_light_status_dim

    def get_lane_coords(self, sample_idx: int) -> FeatureDataType:
        """
        Retrieve lane coordinates at given sample index.
        :param sample_idx: the batch index of interest.
        :return: lane coordinate features.
        """
        lane_coords = self.coords[VectorFeatureLayer.LANE.name][sample_idx]
        if lane_coords.size == 0:
            raise RuntimeError("Lane feature is empty!")
        return lane_coords

    @classmethod
    def collate(cls, batch: List[VectorSetMap]) -> VectorSetMap:
        """Implemented. See interface."""
        coords: Dict[str, List[FeatureDataType]] = defaultdict(list)
        traffic_light_data: Dict[str, List[FeatureDataType]] = defaultdict(list)
        availabilities: Dict[str, List[FeatureDataType]] = defaultdict(list)

        for sample in batch:
            # coords
            for feature_name, feature_coords in sample.coords.items():
                coords[feature_name] += feature_coords

            # traffic light data
            for feature_name, feature_tl_data in sample.traffic_light_data.items():
                traffic_light_data[feature_name] += feature_tl_data

            # availabilities
            for feature_name, feature_avails in sample.availabilities.items():
                availabilities[feature_name] += feature_avails

        return VectorSetMap(coords=coords, traffic_light_data=traffic_light_data, availabilities=availabilities)

    def to_feature_tensor(self) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(
            coords={
                feature_name: [to_tensor(sample).contiguous() for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data={
                feature_name: [to_tensor(sample).contiguous() for sample in feature_tl_data]
                for feature_name, feature_tl_data in self.traffic_light_data.items()
            },
            availabilities={
                feature_name: [to_tensor(sample).contiguous() for sample in feature_avails]
                for feature_name, feature_avails in self.availabilities.items()
            },
        )

    def to_device(self, device: torch.device) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(
            coords={
                feature_name: [sample.to(device=device) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data={
                feature_name: [sample.to(device=device) for sample in feature_tl_data]
                for feature_name, feature_tl_data in self.traffic_light_data.items()
            },
            availabilities={
                feature_name: [sample.to(device=device) for sample in feature_avails]
                for feature_name, feature_avails in self.availabilities.items()
            },
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(
            coords=data["coords"],
            traffic_light_data=data["traffic_light_data"],
            availabilities=data["availabilities"],
        )

    def unpack(self) -> List[VectorSetMap]:
        """Implemented. See interface."""
        return [
            VectorSetMap(
                {feature_name: [feature_coords[sample_idx]] for feature_name, feature_coords in self.coords.items()},
                {
                    feature_name: [feature_tl_data[sample_idx]]
                    for feature_name, feature_tl_data in self.traffic_light_data.items()
                },
                {
                    feature_name: [feature_avails[sample_idx]]
                    for feature_name, feature_avails in self.availabilities.items()
                },
            )
            for sample_idx in range(self.batch_size)
        ]

    def rotate(self, quaternion: Quaternion) -> VectorSetMap:
        """
        Rotate the vector set map.
        :param quaternion: Rotation to apply.
        :return rotated VectorSetMap.
        """
        # Function only works with numpy arrays
        for feature_coords in self.coords.values():
            for sample in feature_coords:
                validate_type(sample, np.ndarray)

        return VectorSetMap(
            coords={
                feature_name: [rotate_coords(sample, quaternion) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )

    def translate(self, translation_value: FeatureDataType) -> VectorSetMap:
        """
        Translate the vector set map.
        :param translation_value: Translation in x, y, z.
        :return translated VectorSetMap.
        :raise ValueError if translation_value dimensions invalid.
        """
        if translation_value.size != 3:
            raise ValueError(
                f"Translation value has incorrect dimensions: {translation_value.size}! Expected: 3 (x, y, z)"
            )
        are_the_same_type(translation_value, list(self.coords.values())[0])

        return VectorSetMap(
            coords={
                feature_name: [
                    translate_coords(sample_coords, translation_value, sample_avails)
                    for sample_coords, sample_avails in zip(
                        self.coords[feature_name], self.availabilities[feature_name]
                    )
                ]
                for feature_name in self.coords
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )

    def scale(self, scale_value: FeatureDataType) -> VectorSetMap:
        """
        Scale the vector set map.
        :param scale_value: <np.float: 3,>. Scale in x, y, z.
        :return scaled VectorSetMap.
        :raise ValueError if scale_value dimensions invalid.
        """
        if scale_value.size != 3:
            raise ValueError(f"Scale value has incorrect dimensions: {scale_value.size}! Expected: 3 (x, y, z)")
        are_the_same_type(scale_value, list(self.coords.values())[0])

        return VectorSetMap(
            coords={
                feature_name: [scale_coords(sample, scale_value) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )

    def xflip(self) -> VectorSetMap:
        """
        Flip the vector set map along the X-axis.
        :return flipped VectorSetMap.
        """
        return VectorSetMap(
            coords={
                feature_name: [xflip_coords(sample) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )

    def yflip(self) -> VectorSetMap:
        """
        Flip the vector set map along the Y-axis.
        :return flipped VectorSetMap.
        """
        return VectorSetMap(
            coords={
                feature_name: [yflip_coords(sample) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )
