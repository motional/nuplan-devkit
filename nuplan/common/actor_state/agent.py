from __future__ import annotations

import copy
from typing import List, Optional

from nuplan.common.actor_state.agent_temporal_state import AgentTemporalState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class Agent(AgentTemporalState, SceneObject):
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
        AgentTemporalState.__init__(
            self,
            initial_time_stamp=TimePoint(metadata.timestamp_us),
            predictions=predictions,
            past_trajectory=past_trajectory,
        )
        SceneObject.__init__(
            self, tracked_object_type=tracked_object_type, oriented_box=oriented_box, metadata=metadata
        )
        self._velocity = velocity
        self._angular_velocity = angular_velocity

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
