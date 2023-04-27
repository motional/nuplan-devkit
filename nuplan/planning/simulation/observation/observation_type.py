from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.database.utils.image import Image
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud


class CameraChannel(Enum):
    """
    An enum class representing supported camera channels
    """

    CAM_F0 = "CAM_F0"
    CAM_B0 = "CAM_B0"
    CAM_L0 = "CAM_L0"
    CAM_L1 = "CAM_L1"
    CAM_L2 = "CAM_L2"
    CAM_R0 = "CAM_R0"
    CAM_R1 = "CAM_R1"
    CAM_R2 = "CAM_R2"


class LidarChannel(Enum):
    """
    An enum class representing supported lidar channels
    """

    MERGED_PC = "MergedPointCloud"


SensorChannel = Union[CameraChannel, LidarChannel]


@dataclass
class Observation(ABC):
    """
    Abstract observation container.
    """

    @classmethod
    def detection_type(cls) -> str:
        """
        Returns detection type of the observation.
        """
        return cls.__name__


@dataclass
class Sensors(Observation):
    """
    Output of sensors, e.g. images or pointclouds.
    """

    pointcloud: Optional[Dict[LidarChannel, LidarPointCloud]]
    images: Optional[Dict[CameraChannel, Image]]


@dataclass
class DetectionsTracks(Observation):
    """
    Output of the perception system, i.e. tracks.
    """

    tracked_objects: TrackedObjects
