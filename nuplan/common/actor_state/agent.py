from __future__ import annotations

import copy
from typing import List, Optional

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class Agent(SceneObject):
    """Class describing Agents in the scene, representing Vehicles, Bicycles and Pedestrians"""

    def __init__(
        self,
        tracked_object_type: TrackedObjectType,
        oriented_box: OrientedBox,
        velocity: StateVector2D,
        metadata: SceneObjectMetadata,
        angular_velocity: Optional[float] = None,
        predictions: Optional[List[PredictedTrajectory]] = None,
        past_trajectory: Optional[PredictedTrajectory] = None,
    ):
        """
        Representation of an Agent in the scene (Vehicles, Pedestrians, Bicyclists and GenericObjects).
        :param tracked_object_type: Type of the current agent
        :param oriented_box: Geometrical representation of the Agent
        :param velocity: Velocity (vectorial) of Agent
        :param metadata: Agent's metadata.
        :param angular_velocity: The scalar angular velocity of the agent, if available
        :param predictions: Optional list of (possibly multiple) predicted trajectories
        """
        super().__init__(tracked_object_type, oriented_box, metadata)
        self._velocity = velocity
        self._angular_velocity = angular_velocity

        # Possible multiple predicted trajectories
        self._predictions: List[PredictedTrajectory] = predictions if predictions is not None else []
        self._past_trajectory = past_trajectory

    @property
    def velocity(self) -> StateVector2D:
        """
        Getter for velocity
        :return: The agent vectorial velocity
        """
        return self._velocity

    @property
    def angular_velocity(self) -> Optional[float]:
        """
        Getter for angular
        :return: The agent angular velocity
        """
        return self._angular_velocity

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
            raise RuntimeError(
                f"Last waypoint represents current agent's {self.metadata.track_id} state, this should not be None!"
            )

        # Sanity check that last waypoint is at the same time index as the current one
        if last_waypoint.time_us != self.metadata.timestamp_us:
            raise ValueError("The provided trajectory does not end at current agent state!")
        self._past_trajectory = past_trajectory

    @classmethod
    def from_new_pose(cls, agent: Agent, pose: StateSE2) -> Agent:
        """
        Initializer that create the same agent in a different pose.
        :param agent: A sample agent
        :param pose: The new pose
        :return: A new agent
        """
        return Agent(
            tracked_object_type=agent.tracked_object_type,
            oriented_box=OrientedBox.from_new_pose(agent.box, pose),
            velocity=agent.velocity,
            angular_velocity=agent.angular_velocity,
            metadata=copy.deepcopy(agent.metadata),
        )
