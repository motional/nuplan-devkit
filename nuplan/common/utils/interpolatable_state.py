from __future__ import annotations

from abc import ABC, abstractmethod

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.utils.split_state import SplitState


class InterpolatableState(ABC):
    """
    Interface for producing interpolatable arrays from objects. Objects must have two methods implemented,
    to_split_state which splits the object state in a list of variables and in a list of constants; and from_split_state
    which constructs a new object using the same set of parameters, a list of variables and a list of constants.
    This is to interpolate states which contain fixed parts.
    """

    @property
    @abstractmethod
    def time_point(self) -> TimePoint:
        """
        Interpolation time
        :return: The time corresponding to the time_point
        """
        pass

    @property
    def time_us(self) -> int:
        """
        Interpolation time
        :return: The time corresponding to the state time point
        """
        return int(self.time_point.time_us)

    @abstractmethod
    def to_split_state(self) -> SplitState:
        """
        Serializes the object in three lists, one containing variable (interpolatable) states, the other
        containing states which are not meant to be interpolated, but are required to de-serialize the object
        after interpolation.
        :return: A tuple with list of variable and a list of fixed states
        """
        pass

    @staticmethod
    @abstractmethod
    def from_split_state(split_state: SplitState) -> InterpolatableState:
        """
        De-serializes an object by its variable and fixed states, for example after interpolation.
        :param split_state: The split state representation
        :return: The deserialized Object
        """
        pass
