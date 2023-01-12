from abc import ABCMeta, abstractmethod
from typing import Any, List, Union

from nuplan.common.actor_state.state_representation import TimePoint


class AbstractTrajectory(metaclass=ABCMeta):
    """
    Generic agent or ego trajectory interface.
    """

    @property
    @abstractmethod
    def start_time(self) -> TimePoint:
        """
        Get the trajectory start time.
        :return: Start time.
        """
        pass

    @property
    @abstractmethod
    def end_time(self) -> TimePoint:
        """
        Get the trajectory end time.
        :return: End time.
        """
        pass

    @property
    def duration(self) -> float:
        """
        :return: the time duration of the trajectory
        """
        return self.end_time.time_s - self.start_time.time_s  # type: ignore

    @property
    def duration_us(self) -> int:
        """
        :return: the time duration of the trajectory in micro seconds
        """
        return int(self.end_time.time_us - self.start_time.time_us)

    @abstractmethod
    def get_state_at_time(self, time_point: TimePoint) -> Any:
        """
        Get the state of the actor at the specified time point.
        :param time_point: Time for which are want to query a state.
        :return: State at the specified time.

        :raises AssertionError: Throws an exception in case a time_point is beyond range of a trajectory.
        """
        pass

    @abstractmethod
    def get_state_at_times(self, time_points: List[TimePoint]) -> List[Any]:
        """
        Get the state of the actor at the specified time points.
        :param time_points: List of time points for which are want to query a state.
        :return: States at the specified time.

        :raises AssertionError: Throws an exception in case a time_point is beyond range of a trajectory.
        """
        pass

    @abstractmethod
    def get_sampled_trajectory(self) -> List[Any]:
        """
        Get the sampled states along the trajectory.
        :return: Discrete trajectory consisting of states.
        """
        pass

    def is_in_range(self, time_point: Union[TimePoint, int]) -> bool:
        """
        Check whether a time point is in range of trajectory.
        :return: True if it is, False otherwise.
        """
        if isinstance(time_point, int):
            time_point = TimePoint(time_point)
        return bool(self.start_time <= time_point <= self.end_time)
