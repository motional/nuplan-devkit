from __future__ import annotations

from typing import Any, Iterable, List, Optional, Union

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.utils.split_state import SplitState


class Waypoint(InterpolatableState):
    """Represents a waypoint which is part of a trajectory. Optionals to allow for geometric trajectory"""

    def __init__(self, time_point: TimePoint, oriented_box: OrientedBox, velocity: Optional[StateVector2D] = None):
        """
        :param time_point: TimePoint corresponding to the Waypoint
        :param oriented_box: Position of the oriented box at the Waypoint
        :param velocity: Optional velocity information
        """
        self._time_point = time_point
        self._oriented_box = oriented_box
        self._velocity = velocity

    def __iter__(self) -> Iterable[Union[int, float]]:
        """
        Iterator for waypoint variables.
        :return: An iterator to the variables of the Waypoint.
        """
        return iter(
            (
                self.time_us,
                self._oriented_box.center.x,
                self._oriented_box.center.y,
                self._oriented_box.center.heading,
                self._velocity.x if self._velocity is not None else None,
                self._velocity.y if self._velocity is not None else None,
            )
        )

    def __eq__(self, other: Any) -> bool:
        """
        Comparison between two Waypoints.
        :param other: Other object.
        :return True if both objects are same.
        """
        if not isinstance(other, Waypoint):
            return NotImplemented

        return (
            other.oriented_box == self._oriented_box
            and other.time_point == self.time_point
            and other.velocity == self._velocity
        )

    def __repr__(self) -> str:
        """
        :return: A string describing the object.
        """
        return self.__class__.__qualname__ + "(" + ', '.join([f"{f}={v}" for f, v in self.__dict__.items()]) + ")"

    @property
    def center(self) -> StateSE2:
        """
        Getter for center position of the waypoint
        :return: StateSE2 referring to position of the waypoint
        """
        return self._oriented_box.center

    @property
    def time_point(self) -> TimePoint:
        """
        Getter for time point corresponding to the waypoint
        :return: The time point
        """
        return self._time_point

    @property
    def oriented_box(self) -> OrientedBox:
        """
        Getter for the oriented box corresponding to the waypoint
        :return: The oriented box
        """
        return self._oriented_box

    @property
    def x(self) -> float:
        """
        Getter for the x position of the waypoint
        :return: The x position
        """
        return self._oriented_box.center.x  # type:ignore

    @property
    def y(self) -> float:
        """
        Getter for the y position of the waypoint
        :return: The y position
        """
        return self._oriented_box.center.y  # type:ignore

    @property
    def heading(self) -> float:
        """
        Getter for the heading of the waypoint
        :return: The heading
        """
        return self._oriented_box.center.heading  # type:ignore

    @property
    def velocity(self) -> Optional[StateVector2D]:
        """
        Getter for the velocity corresponding to the waypoint
        :return: The velocity, None if not available
        """
        return self._velocity

    def serialize(self) -> List[Union[int, float]]:
        """
        Serializes the object as a list
        :return: Serialized object as a list
        """
        return [
            self.time_point.time_us,
            self._oriented_box.center.x,
            self._oriented_box.center.y,
            self._oriented_box.center.heading,
            self._oriented_box.length,
            self._oriented_box.width,
            self._oriented_box.height,
            self._velocity.x if self._velocity is not None else None,
            self._velocity.y if self._velocity is not None else None,
        ]

    @staticmethod
    def deserialize(vector: List[Union[int, float]]) -> Waypoint:
        """
        Deserializes the object.
        :param vector: a list of data to initialize a waypoint
        :return: Waypoint
        """
        assert len(vector) == 9, f'Expected a vector of size 9, got {len(vector)}'

        return Waypoint(
            time_point=TimePoint(int(vector[0])),
            oriented_box=OrientedBox(StateSE2(vector[1], vector[2], vector[3]), vector[4], vector[5], vector[6]),
            velocity=StateVector2D(vector[7], vector[8]) if vector[7] is not None and vector[8] is not None else None,
        )

    def to_split_state(self) -> SplitState:
        """Inherited, see superclass."""
        linear_states = [
            self.time_point.time_us,
            self._oriented_box.center.x,
            self._oriented_box.center.y,
            self._velocity.x if self._velocity is not None else None,
            self._velocity.y if self._velocity is not None else None,
        ]
        angular_states = [self._oriented_box.center.heading]
        fixed_state = [self._oriented_box.width, self._oriented_box.length, self._oriented_box.height]

        return SplitState(linear_states, angular_states, fixed_state)

    @staticmethod
    def from_split_state(split_state: SplitState) -> Waypoint:
        """Inherited, see superclass."""
        total_state_length = len(split_state)

        assert total_state_length == 9, f'Expected a vector of size 9, got {total_state_length}'

        return Waypoint(
            time_point=TimePoint(int(split_state.linear_states[0])),
            oriented_box=OrientedBox(
                StateSE2(split_state.linear_states[1], split_state.linear_states[2], split_state.angular_states[0]),
                length=split_state.fixed_states[1],
                width=split_state.fixed_states[0],
                height=split_state.fixed_states[2],
            ),
            velocity=StateVector2D(split_state.linear_states[3], split_state.linear_states[4])
            if split_state.linear_states[3] is not None and split_state.linear_states[4] is not None
            else None,
        )
