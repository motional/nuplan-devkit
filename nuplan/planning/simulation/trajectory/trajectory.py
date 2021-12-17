from abc import ABCMeta, abstractmethod
from typing import List, Union

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
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

    @abstractmethod
    def get_state_at_time(self, time_point: TimePoint) -> Union[EgoState, Agent]:
        """
        Get the state of the actor at the specified time point.
        :param time_point: Time for which are want to query a state.
        :return: State at the specified time.

        :raises Exception: Throws an exception in case a time_point is beyond range of a trajectory.
        """
        pass

    @abstractmethod
    def get_sampled_trajectory(self) -> List[Union[EgoState, Agent]]:
        """
        Get the sampled states along the trajectory.
        :return: Discrete trajectory consisting of states.
        """
        pass
