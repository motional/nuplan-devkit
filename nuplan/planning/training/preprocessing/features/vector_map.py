from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List

import numpy as np
import torch
from pyquaternion import Quaternion

from nuplan.planning.script.builders.utils.utils_type import are_the_same_type, validate_type
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import LaneOnRouteStatusData
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
class VectorMap(AbstractModelFeature):
    """
    Vector map data struture, including:
        coords: List[<np.ndarray: num_lane_segments, 2, 2>].
            The (x, y) coordinates of the start and end point of the lane segments.
        lane_groupings: List[List[<np.ndarray: num_lane_segments_in_lane>]].
            Each lane grouping or polyline is represented by an array of indices of lane segments
            in coords belonging to the given lane. Each batch contains a List of lane groupings.
        multi_scale_connections: List[Dict of {scale: connections_of_scale}].
            Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
            and each column in the array is [from_lane_segment_idx, to_lane_segment_idx].
        on_route_status: List[<np.ndarray: num_lane_segments, 2>].
            Binary encoding of on route status for lane segment at given index.
            Encoding: off route [0, 1], on route [1, 0], unknown [0, 0]
        traffic_light_data: List[<np.ndarray: num_lane_segments, 4>]
            One-hot encoding of on traffic light status for lane segment at given index.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]

    In all cases, the top level List represent number of batches. This is a special feature where
    each batch entry can have different size. Similarly, each lane grouping within a batch can have
    a variable number of elements. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """

    coords: List[FeatureDataType]
    lane_groupings: List[List[FeatureDataType]]
    multi_scale_connections: List[Dict[int, FeatureDataType]]
    on_route_status: List[FeatureDataType]
    traffic_light_data: List[FeatureDataType]
    _lane_coord_dim: int = 2
    _on_route_status_encoding_dim: int = LaneOnRouteStatusData.encoding_dim()

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        if len(self.coords) != len(self.multi_scale_connections):
            raise RuntimeError(
                f"Not consistent length of batches! {len(self.coords)} != {len(self.multi_scale_connections)}"
            )

        if len(self.coords) != len(self.lane_groupings):
            raise RuntimeError(f"Not consistent length of batches! {len(self.coords)} != {len(self.lane_groupings)}")

        if len(self.coords) != len(self.on_route_status):
            raise RuntimeError(f"Not consistent length of batches! {len(self.coords)} != {len(self.on_route_status)}")

        if len(self.coords) != len(self.traffic_light_data):
            raise RuntimeError(
                f"Not consistent length of batches! {len(self.coords)} != {len(self.traffic_light_data)}"
            )

        if len(self.coords) == 0:
            raise RuntimeError("Batch size has to be > 0!")

        for coords in self.coords:
            if coords.shape[1] != 2 or coords.shape[2] != 2:
                raise RuntimeError("The dimension of coords is not correct!")

        for coords, traffic_lights in zip(self.coords, self.traffic_light_data):
            if coords.shape[0] != traffic_lights.shape[0]:
                raise RuntimeError("Number of segments are inconsistent")

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return (
            len(self.coords) > 0
            and len(self.coords[0]) > 0
            and len(self.lane_groupings) > 0
            and len(self.lane_groupings[0]) > 0
            and len(self.lane_groupings[0][0]) > 0
            and len(self.on_route_status) > 0
            and len(self.on_route_status[0]) > 0
            and len(self.traffic_light_data) > 0
            and len(self.traffic_light_data[0]) > 0
            and len(self.multi_scale_connections) > 0
            and len(list(self.multi_scale_connections[0].values())[0]) > 0
        )

    @property
    def num_of_batches(self) -> int:
        """
        :return: number of batches
        """
        return len(self.coords)

    def num_lanes_in_sample(self, sample_idx: int) -> int:
        """
        :param sample_idx: sample index in batch
        :return: number of lanes represented by lane_groupings in sample
        """
        return len(self.lane_groupings[sample_idx])

    @classmethod
    def lane_coord_dim(cls) -> int:
        """
        :return: dimension of coords, should be 2 (x, y)
        """
        return cls._lane_coord_dim

    @classmethod
    def on_route_status_encoding_dim(cls) -> int:
        """
        :return: dimension of route following status encoding
        """
        return cls._on_route_status_encoding_dim

    @classmethod
    def flatten_lane_coord_dim(cls) -> int:
        """
        :return: dimension of flattened start and end coords, should be 4 = 2 x (x, y)
        """
        return 2 * cls._lane_coord_dim

    def get_lane_coords(self, sample_idx: int) -> FeatureDataType:
        """
        Retrieve lane coordinates at given sample index.
        :param sample_idx: the batch index of interest.
        :return: lane coordinate features.
        """
        return self.coords[sample_idx]

    @classmethod
    def collate(cls, batch: List[VectorMap]) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(
            coords=[data for sample in batch for data in sample.coords],
            lane_groupings=[data for sample in batch for data in sample.lane_groupings],
            multi_scale_connections=[data for sample in batch for data in sample.multi_scale_connections],
            on_route_status=[data for sample in batch for data in sample.on_route_status],
            traffic_light_data=[data for sample in batch for data in sample.traffic_light_data],
        )

    def to_feature_tensor(self) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(
            coords=[to_tensor(coords).contiguous() for coords in self.coords],
            lane_groupings=[
                [to_tensor(lane_grouping).contiguous() for lane_grouping in lane_groupings]
                for lane_groupings in self.lane_groupings
            ],
            multi_scale_connections=[
                {scale: to_tensor(connection).contiguous() for scale, connection in multi_scale_connections.items()}
                for multi_scale_connections in self.multi_scale_connections
            ],
            on_route_status=[to_tensor(status).contiguous() for status in self.on_route_status],
            traffic_light_data=[to_tensor(data).contiguous() for data in self.traffic_light_data],
        )

    def to_device(self, device: torch.device) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(
            coords=[coords.to(device=device) for coords in self.coords],
            lane_groupings=[
                [lane_grouping.to(device=device) for lane_grouping in lane_groupings]
                for lane_groupings in self.lane_groupings
            ],
            multi_scale_connections=[
                {scale: connection.to(device=device) for scale, connection in multi_scale_connections.items()}
                for multi_scale_connections in self.multi_scale_connections
            ],
            on_route_status=[status.to(device=device) for status in self.on_route_status],
            traffic_light_data=[data.to(device=device) for data in self.traffic_light_data],
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VectorMap:
        """Implemented. See interface."""
        return VectorMap(
            coords=data["coords"],
            lane_groupings=data["lane_groupings"],
            multi_scale_connections=data["multi_scale_connections"],
            on_route_status=data["on_route_status"],
            traffic_light_data=data["traffic_light_data"],
        )

    def unpack(self) -> List[VectorMap]:
        """Implemented. See interface."""
        return [
            VectorMap([coords], [lane_groupings], [multi_scale_connections], [on_route_status], [traffic_light_data])
            for coords, lane_groupings, multi_scale_connections, on_route_status, traffic_light_data in zip(
                self.coords,
                self.lane_groupings,
                self.multi_scale_connections,
                self.on_route_status,
                self.traffic_light_data,
            )
        ]

    def rotate(self, quaternion: Quaternion) -> VectorMap:
        """
        Rotate the vector map.
        :param quaternion: Rotation to apply.
        """
        # Function only works with numpy arrays
        for coord in self.coords:
            validate_type(coord, np.ndarray)

        return VectorMap(
            coords=[rotate_coords(data, quaternion) for data in self.coords],
            lane_groupings=self.lane_groupings,
            multi_scale_connections=self.multi_scale_connections,
            on_route_status=self.on_route_status,
            traffic_light_data=self.traffic_light_data,
        )

    def translate(self, translation_value: FeatureDataType) -> VectorMap:
        """
        Translate the vector map.
        :param translation_value: Translation in x, y, z.
        """
        assert translation_value.size == 3, "Translation value must have dimension of 3 (x, y, z)"
        are_the_same_type(translation_value, self.coords[0])

        return VectorMap(
            coords=[translate_coords(coords, translation_value) for coords in self.coords],
            lane_groupings=self.lane_groupings,
            multi_scale_connections=self.multi_scale_connections,
            on_route_status=self.on_route_status,
            traffic_light_data=self.traffic_light_data,
        )

    def scale(self, scale_value: FeatureDataType) -> VectorMap:
        """
        Scale the vector map.
        :param scale_value: <np.float: 3,>. Scale in x, y, z.
        """
        assert scale_value.size == 3, f"Scale value has incorrect dimension: {scale_value.size}!"
        are_the_same_type(scale_value, self.coords[0])

        return VectorMap(
            coords=[scale_coords(coords, scale_value) for coords in self.coords],
            lane_groupings=self.lane_groupings,
            multi_scale_connections=self.multi_scale_connections,
            on_route_status=self.on_route_status,
            traffic_light_data=self.traffic_light_data,
        )

    def xflip(self) -> VectorMap:
        """
        Flip the vector map along the X-axis.
        """
        return VectorMap(
            coords=[xflip_coords(coords) for coords in self.coords],
            lane_groupings=self.lane_groupings,
            multi_scale_connections=self.multi_scale_connections,
            on_route_status=self.on_route_status,
            traffic_light_data=self.traffic_light_data,
        )

    def yflip(self) -> VectorMap:
        """
        Flip the vector map along the Y-axis.
        """
        return VectorMap(
            coords=[yflip_coords(coords) for coords in self.coords],
            lane_groupings=self.lane_groupings,
            multi_scale_connections=self.multi_scale_connections,
            on_route_status=self.on_route_status,
            traffic_light_data=self.traffic_light_data,
        )

    def extract_lane_polyline(self, sample_idx: int, lane_idx: int) -> FeatureDataType:
        """
        Extract start points (first coordinate) for segments in lane, specified by segment indices
            in lane_groupings.
        :param sample_idx: sample index in batch
        :param lane_idx: lane index in sample
        :return: lane_polyline: <np.ndarray: num_lane_segments_in_lane, 2>. Array of start points
            for each segment in lane.
        """
        lane_grouping = self.lane_groupings[sample_idx][lane_idx]
        return self.coords[sample_idx][lane_grouping, 0]
