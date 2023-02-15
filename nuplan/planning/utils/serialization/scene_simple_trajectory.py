from typing import Any, Dict, List

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class SceneSimpleTrajectory(AbstractTrajectory):
    """
    Simple trajectory that is used to represent scene's predictions.
    """

    def __init__(self, prediction_states: List[Dict[str, Any]], width: float, length: float, height: float):
        """
        Constructor.

        :param prediction_states: Dictionary of states.
        :param width: [m] Width of the agent.
        :param length: [m] Length of the agent.
        :param height: [m] Height of the agent.
        """
        self._states: List[Waypoint] = []
        self._state_at_time: Dict[TimePoint, Waypoint] = {}

        for state in prediction_states:
            time = TimePoint(int(state["timestamp"] * 1e6))  # in json, the states are in seconds
            coordinates: List[float] = state["pose"]
            center = StateSE2(x=coordinates[0], y=coordinates[1], heading=coordinates[2])
            self._states.append(
                Waypoint(
                    time_point=time, oriented_box=OrientedBox(center=center, width=width, length=length, height=height)
                )
            )
            self._state_at_time[time] = self._states[-1]

        self._start_time = prediction_states[0]["timestamp"]
        self._end_time = prediction_states[-1]["timestamp"]

    @property
    def start_time(self) -> TimePoint:
        """
        Get the trajectory start time.
        :return: Start time.
        """
        return self._start_time

    @property
    def end_time(self) -> TimePoint:
        """
        Get the trajectory end time.
        :return: End time.
        """
        return self._end_time

    def get_state_at_time(self, time_point: TimePoint) -> Any:
        """
        Get the state of the actor at the specified time point.
        :param time_point: Time for which are want to query a state.
        :return: State at the specified time.

        :raises Exception: Throws an exception in case a time_point is beyond range of a trajectory.
        """
        return self._state_at_time[time_point]

    def get_state_at_times(self, time_points: List[TimePoint]) -> List[Any]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_sampled_trajectory(self) -> List[Any]:
        """
        Get the sampled states along the trajectory.
        :return: Discrete trajectory consisting of states.
        """
        return self._states
