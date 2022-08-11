from __future__ import annotations

from typing import List, Optional

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class AgentTemporalState:
    """
    Actor with current, multimodal future as well as past trajectory.
        The future trajectory probabilities have to sum up to 1.0.
        The past trajectory is only single modal with mode probability 1.0.
        The last waypoint in past trajectory has to be the same as current position (we check only timestamp).
    """

    def __init__(
        self,
        initial_time_stamp: TimePoint,
        predictions: Optional[List[PredictedTrajectory]] = None,
        past_trajectory: Optional[PredictedTrajectory] = None,
    ):
        """
        Initialize actor temporal state which has past as well as future trajectory
        :param initial_time_stamp: time stamp the current detections
        :param predictions: future multimodal trajectory
        :param past_trajectory: past trajectory transversed
        """
        self._initial_time_stamp = initial_time_stamp
        self.predictions: List[PredictedTrajectory] = predictions if predictions is not None else []
        self.past_trajectory = past_trajectory

    @property
    def previous_state(self) -> Optional[Waypoint]:
        """
        :return: None if agent's previous state does not exists, otherwise return previous state
        """
        # At minimum 2 states are required since the last state is the same as current state
        if not self.past_trajectory or len(self.past_trajectory.valid_waypoints) < 2:
            return None
        return self.past_trajectory.waypoints[-2]

    @property
    def predictions(self) -> List[PredictedTrajectory]:
        """
        Getter for agents predicted trajectories
        :return: Trajectories
        """
        return self._predictions

    @predictions.setter
    def predictions(self, predicted_trajectories: List[PredictedTrajectory]) -> None:
        """
        Setter for predicted trajectories, checks if the listed probabilities sum to one.
        :param predicted_trajectories: List of Predicted trajectories
        """
        if not predicted_trajectories:
            self._predictions = predicted_trajectories
            return
        # Sanity check that if predictions are provided, probabilities sum to 1
        probability_sum = sum(prediction.probability for prediction in predicted_trajectories)
        if not abs(probability_sum - 1) < 1e-6 and predicted_trajectories:
            raise ValueError(f"The provided trajectory probabilities did not sum to one, but to {probability_sum:.2f}!")
        self._predictions = predicted_trajectories

    @property
    def past_trajectory(self) -> Optional[PredictedTrajectory]:
        """
        Getter for agents predicted trajectories
        :return: Trajectories
        """
        return self._past_trajectory

    @past_trajectory.setter
    def past_trajectory(self, past_trajectory: Optional[PredictedTrajectory]) -> None:
        """
        Setter for predicted trajectories, checks if the listed probabilities sum to one.
        :param past_trajectory: Driven Trajectory
        """
        if not past_trajectory:
            # In case it is none, no check is needed
            self._past_trajectory = past_trajectory
            return

        # Make sure that the current state is set!
        last_waypoint = past_trajectory.waypoints[-1]
        if not last_waypoint:
            raise RuntimeError("Last waypoint represents current agent's state, this should not be None!")

        # Sanity check that last waypoint is at the same time index as the current one
        if last_waypoint.time_point != self._initial_time_stamp:
            raise ValueError(
                "The provided trajectory does not end at current agent state!"
                f" {last_waypoint.time_us} != {self._initial_time_stamp}"
            )
        self._past_trajectory = past_trajectory
