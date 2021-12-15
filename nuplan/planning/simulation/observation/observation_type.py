from abc import ABC
from dataclasses import dataclass
from typing import List

from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.maps_datatypes import PointCloud
from nuplan.database.utils.boxes.box3d import Box3D


@dataclass
class Observation(ABC):
    """
    Abstract observation container.
    """

    @classmethod
    def detection_type(cls) -> str:
        return cls.__name__


@dataclass
class Sensors(Observation):
    """
    Output of sensors, e.g. images or pointclouds.
    """
    pointcloud: PointCloud


@dataclass
class Detections(Observation):
    """
    Output of the perception system, i.e. tracks.
    """
    boxes: List[Box3D]


@dataclass
class DetectionsTracks(Observation):
    """
    Output of the perception system, i.e. tracks.
    """
    tracked_objects: TrackedObjects
