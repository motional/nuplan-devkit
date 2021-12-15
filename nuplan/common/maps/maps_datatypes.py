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

    @classmethod
    def deserialize(cls, layer: str) -> SemanticMapLayer:
        """ Deserialize the type when loading from a string. """
        return SemanticMapLayer.__members__[layer]


class StopLineType(IntEnum):
    PED_CROSSING = 0
    STOP_SIGN = 1
    TRAFFIC_LIGHT = 2
    TURN_STOP = 3
    YIELD = 4


class IntersectionType(IntEnum):
    DEFAULT = 0
    TRAFFIC_LIGHT = 1
    STOP_SIGN = 2
    LANE_BRANCH = 3
    LANE_MERGE = 4
    PASS_THROUGH = 5


class TrafficLightStatusType(IntEnum):
    GREEN = 0
    YELLOW = 1
    RED = 2

    def serialize(self) -> str:
        """ Serialize the type when saving. """

        return self.name

    @classmethod
    def deserialize(cls, key: str) -> TrafficLightStatusType:
        """ Deserialize the type when loading from a string. """

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
    layers: Dict[SemanticMapLayer, VectorLayer]  # type: ignore


@dataclass
class RasterMap:
    layers: Dict[SemanticMapLayer, RasterLayer]


@dataclass
class LaneSegmentCoords:
    coords: List[Tuple[Point2D, Point2D]]  # lane-segment coordinates in format of [N, 2, 2] representing
    # [num_lane_segment, [start coords, end coords]]

    def to_vector(self) -> List[List[List[float]]]:
        """
        Returns data in vectorized form
        :return: vectorized lane segment coordinates in [num_lane_segment, 2, 2]
        """
        return [[[start.x, start.y], [end.x, end.y]] for start, end in self.coords]


@dataclass
class LaneSegmentConnections:
    connections: List[Tuple[int, int]]  # lane-segment connection relations in format of [num_connection, 2]
    # and each column in the array is (from_lane_segment_idx, to_lane_segment_idx)

    def to_vector(self) -> List[List[int]]:
        """
        Returns data in vectorized form
        :return: vectorized lane segment connections as [num_lane_segment, 2, 2]
        """
        return [[start, end] for start, end in self.connections]


@dataclass
class LaneSegmentMetaData:
    """ Place holder for when prediction team implements this """
    data: List[List[Any]]  # lane_segment meta data info including in_intersection, turn_direction,etc


@dataclass
class TrafficLightStatusData:
    """ Traffic light status. """

    status: TrafficLightStatusType  # Status: green, red
    stop_line_id: int  # Stop line id, where this traffic light belongs to
    lane_connector_id: int  # lane connector id, where this traffic light belongs to
    timestamp: int  # Timestamp

    def serialize(self) -> Dict[str, Any]:
        """ Serialize traffic light status. """

        return {
            'status': self.status.serialize(),
            'stop_line_id': self.stop_line_id,
            'lane_connector_id': self.lane_connector_id,
            'timestamp': self.timestamp
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TrafficLightStatusData:
        """ Deserialize a dict of data to this class. """

        return TrafficLightStatusData(status=TrafficLightStatusType.deserialize(data['status']),
                                      stop_line_id=data['stop_line_id'],
                                      lane_connector_id=data['lane_connector_id'],
                                      timestamp=data['timestamp']
                                      )
