from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import Point2D

Transform = npt.NDArray[np.float32]  # 4x4 homogeneous transformation matrix

PointCloud = npt.NDArray[np.float32]  # Nx4 array of lidar points (TODO: wrap in dataclass)

VectorLayer = gpd.GeoDataFrame


class SemanticMapLayer(IntEnum):
    """
    Enum for SemanticMapLayers.
    """

    LANE = 0
    INTERSECTION = 1
    STOP_LINE = 2
    TURN_STOP = 3
    CROSSWALK = 4
    DRIVABLE_AREA = 5
    YIELD = 6
    TRAFFIC_LIGHT = 7
    STOP_SIGN = 8
    EXTENDED_PUDO = 9
    SPEED_BUMP = 10
    LANE_CONNECTOR = 11
    BASELINE_PATHS = 12
    WALKWAYS = 13
    CARPARK_AREA = 14
    PUDO = 15
    ROADBLOCK = 16
    ROADBLOCK_CONNECTOR = 17

    @classmethod
    def deserialize(cls, layer: str) -> SemanticMapLayer:
        """Deserialize the type when loading from a string."""
        return SemanticMapLayer.__members__[layer]


class StopLineType(IntEnum):
    """
    Enum for StopLineType.
    """

    PED_CROSSING = 0
    STOP_SIGN = 1
    TRAFFIC_LIGHT = 2
    TURN_STOP = 3
    YIELD = 4


class IntersectionType(IntEnum):
    """
    Enum for IntersectionType.
    """

    DEFAULT = 0
    TRAFFIC_LIGHT = 1
    STOP_SIGN = 2
    LANE_BRANCH = 3
    LANE_MERGE = 4
    PASS_THROUGH = 5


class OnRouteStatusType(IntEnum):
    """
    Enum for OnRouteStatusType.
    """

    OFF_ROUTE = 0
    ON_ROUTE = 1
    UNKNOWN = 2


class TrafficLightStatusType(IntEnum):
    """
    Enum for TrafficLightStatusType.
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    UNKNOWN = 3

    def serialize(self) -> str:
        """Serialize the type when saving."""
        return self.name

    @classmethod
    def deserialize(cls, key: str) -> TrafficLightStatusType:
        """Deserialize the type when loading from a string."""
        return TrafficLightStatusType.__members__[key]


@dataclass
class RasterLayer:
    """
    Wrapper dataclass of a layer of the rasterized map.
    """

    data: npt.NDArray[np.uint8]  # raster image as numpy array
    precision: np.float64  # [m] presision of map
    transform: Transform  # transform from physical to pixel coordinates


@dataclass
class VectorMap:
    """
    Dataclass mapping SemanticMapLayers to associated VectorLayer.
    """

    layers: Dict[SemanticMapLayer, VectorLayer]  # type: ignore


@dataclass
class RasterMap:
    """
    Dataclass mapping SemanticMapLayers to associated RasterLayer.
    """

    layers: Dict[SemanticMapLayer, RasterLayer]


@dataclass
class LaneSegmentCoords:
    """
    Lane-segment coordinates in format of [N, 2, 2] representing [num_lane_segment, [start coords, end coords]].
    """

    coords: List[Tuple[Point2D, Point2D]]

    def to_vector(self) -> List[List[List[float]]]:
        """
        Returns data in vectorized form.
        :return: vectorized lane segment coordinates in [num_lane_segment, 2, 2].
        """
        return [[[start.x, start.y], [end.x, end.y]] for start, end in self.coords]


@dataclass
class LaneSegmentConnections:
    """
    Lane-segment connection relations in format of [num_connection, 2] and each column in the array is
    (from_lane_segment_idx, to_lane_segment_idx).
    """

    connections: List[Tuple[int, int]]

    def to_vector(self) -> List[List[int]]:
        """
        Returns data in vectorized form.
        :return: vectorized lane segment connections as [num_lane_segment, 2, 2].
        """
        return [[start, end] for start, end in self.connections]


@dataclass
class LaneSegmentGroupings:
    """
    Lane-segment groupings in format of [num_lane, num_segment_in_lane (variable size)]
    containing a list of indices of lane segments in corresponding coords list for each lane.
    """

    groupings: List[List[int]]

    def to_vector(self) -> List[List[int]]:
        """
        Returns data in vectorized form.
        :return: vectorized groupings of lane segments as [num_lane, num_lane_segment_in_lane].
        """
        return [[segment_id for segment_id in grouping] for grouping in self.groupings]


@dataclass
class LaneSegmentLaneIDs:
    """
    IDs of lane/lane connectors that lane segment at specified index belong to.
    """

    lane_ids: List[str]


@dataclass
class LaneSegmentRoadBlockIDs:
    """
    IDs of roadblock/roadblock connectors that lane segment at specified index belong to.
    """

    roadblock_ids: List[str]


@dataclass
class LaneOnRouteStatusData:
    """
    Route following status data represented as binary encoding per lane segment [num_lane_segment, 2].
    The binary encoding: off route [0, 1], on route [1, 0], unknown [0, 0].
    """

    on_route_status: List[Tuple[int, int]]

    _binary_encoding = {
        OnRouteStatusType.OFF_ROUTE: (0, 1),
        OnRouteStatusType.ON_ROUTE: (1, 0),
        OnRouteStatusType.UNKNOWN: (0, 0),
    }

    def to_vector(self) -> List[List[float]]:
        """
        Returns data in vectorized form.
        :return: vectorized on route status data per lane segment as [num_lane_segment, 2].
        """
        return [list(data) for data in self.on_route_status]

    @classmethod
    def encode(cls, on_route_status_type: OnRouteStatusType) -> Tuple[int, int]:
        """
        Binary encoding of OnRouteStatusType: off route [0, 0], on route [0, 1], unknown [1, 0].
        """
        return cls._binary_encoding[on_route_status_type]


@dataclass
class LaneSegmentTrafficLightData:
    """
    Traffic light data represented as one-hot encoding per segment [num_lane_segment, 4].
    The one-hot encoding: green [1, 0, 0, 0], yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1].
    """

    traffic_lights: List[Tuple[int, int, int, int]]

    _one_hot_encoding = {
        TrafficLightStatusType.GREEN: (1, 0, 0, 0),
        TrafficLightStatusType.YELLOW: (0, 1, 0, 0),
        TrafficLightStatusType.RED: (0, 0, 1, 0),
        TrafficLightStatusType.UNKNOWN: (0, 0, 0, 1),
    }

    def to_vector(self) -> List[List[float]]:
        """
        Returns data in vectorized form.
        :return: vectorized traffic light data per segment as [num_lane_segment, 3].
        """
        return [list(data) for data in self.traffic_lights]

    @classmethod
    def encode(cls, traffic_light_type: TrafficLightStatusType) -> Tuple[int, int, int, int]:
        """
        One-hot encoding of TrafficLightStatusType: green [1, 0, 0, 0], yellow [0, 1, 0, 0], red [0, 0, 1, 0],
            unknown [0, 0, 0, 1].
        """
        return cls._one_hot_encoding[traffic_light_type]


@dataclass
class TrafficLightStatusData:
    """Traffic light status."""

    status: TrafficLightStatusType  # Status: green, red
    lane_connector_id: int  # lane connector id, where this traffic light belongs to
    timestamp: int  # Timestamp

    def serialize(self) -> Dict[str, Any]:
        """Serialize traffic light status."""
        return {
            'status': self.status.serialize(),
            'lane_connector_id': self.lane_connector_id,
            'timestamp': self.timestamp,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TrafficLightStatusData:
        """Deserialize a dict of data to this class."""
        return TrafficLightStatusData(
            status=TrafficLightStatusType.deserialize(data['status']),
            lane_connector_id=data['lane_connector_id'],
            timestamp=data['timestamp'],
        )
