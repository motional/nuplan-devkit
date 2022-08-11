from __future__ import annotations

from typing import List, Optional

from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.agent_temporal_state import AgentTemporalState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class Agent(AgentTemporalState, AgentState):
    """
    AgentState with future and past trajectory.
    """

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
        :param tracked_object_type: Type of the current agent.
        :param oriented_box: Geometrical representation of the Agent.
        :param velocity: Velocity (vectorial) of Agent.
        :param metadata: Agent's metadata.
        :param angular_velocity: The scalar angular velocity of the agent, if available.
        :param predictions: Optional list of (possibly multiple) predicted trajectories.
        :param past_trajectory: Optional past trajectory of this agent.
        """
        AgentTemporalState.__init__(
            self,
            initial_time_stamp=TimePoint(metadata.timestamp_us),
            predictions=predictions,
            past_trajectory=past_trajectory,
        )
        AgentState.__init__(
            self,
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            metadata=metadata,
            velocity=velocity,
            angular_velocity=angular_velocity,
        )

    @classmethod
    def from_agent_state(cls, agent: AgentState) -> Agent:
        """
        Create Agent from AgentState.
        :param agent: input single agent state.
        :return: Agent with None for future and past trajectory.
        """
        return cls(
            tracked_object_type=agent.tracked_object_type,
            oriented_box=agent.box,
            velocity=agent.velocity,
            metadata=agent.metadata,
            angular_velocity=agent.angular_velocity,
            predictions=None,
            past_trajectory=None,
        )
