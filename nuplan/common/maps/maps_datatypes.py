from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import numpy.typing as npt

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
    BOUNDARIES = 13
    WALKWAYS = 14
    CARPARK_AREA = 15
    PUDO = 16
    ROADBLOCK = 17
    ROADBLOCK_CONNECTOR = 18

    @classmethod
    def deserialize(cls, layer: str) -> SemanticMapLayer:
        """Deserialize the type when loading from a string."""
        return SemanticMapLayer.__members__[layer]


class LaneConnectorType(IntEnum):
    """
    Enum for LaneConnectorType.
    """

    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    UTURN = 3
    UNKNOWN = 4


class StopLineType(IntEnum):
    """
    Enum for StopLineType.
    """

    PED_CROSSING = 0
    STOP_SIGN = 1
    TRAFFIC_LIGHT = 2
    TURN_STOP = 3
    YIELD = 4
    UNKNOWN = 5


class PudoType(IntEnum):
    """
    Enum for PudoType
    """

    PICK_UP_DROP_OFF = 0
    PICK_UP_ONLY = 1
    DROP_OFF_ONLY = 2
    UNKNOWN = 3


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
    precision: np.float64  # [m] precision of map
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


@dataclass
class TrafficLightStatuses:
    """
    Collection of TrafficLightStatusData at a time step.
    """

    traffic_lights: List[TrafficLightStatusData]
