from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.utils.color import Color


class SceneStructure:
    """
    Base class for scene data.
    """

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """
        Iterates through attributes. Used to convert class into dict for serialization.
        Similar to dataclasses.asdict, except it skips some attributes and can convert some of its attributes to dicts too.
        """
        for dim in fields(self):
            value = getattr(self, dim.name)
            if value is None:
                # Attribute is skipped
                continue
            elif issubclass(type(value), SceneStructure):
                # Convert attribute to dictionary before adding
                yield dim.name, dict(value)
            elif isinstance(value, dict):
                # Convert dictionary with scene structures values to dictionary before adding
                yield dim.name, {
                    x: dict(value[x]) if issubclass(type(value[x]), SceneStructure) else value[x] for x in value
                }
            elif isinstance(value, Iterable):
                # Convert any scene structures in iterable attribute to dictionary
                yield dim.name, [dict(v) if issubclass(type(v), SceneStructure) else v for v in value]
            else:
                yield dim.name, value


Pose = Union[StateSE2, List[float]]  # [x, y, heading]


@dataclass
class TrajectoryState(SceneStructure):
    """TrajectoryState format"""

    pose: Pose
    speed: float
    lateral: List[float]  # [dist_left, dist_right]

    velocity_2d: Optional[List[float]] = None  # [x, y]
    acceleration: Optional[List[float]] = None  # [x, y]
    tire_steering_angle: Optional[float] = None


@dataclass
class Trajectory(SceneStructure):
    """Trajectory format"""

    color: Color  # 0-255
    states: List[TrajectoryState]


@dataclass
class GoalScene(SceneStructure):
    """Goal format"""

    pose: Pose


@dataclass
class EgoScene(SceneStructure):
    """Ego format"""

    acceleration: float
    pose: Pose
    speed: float

    prediction: Optional[Dict[str, Any]] = None
