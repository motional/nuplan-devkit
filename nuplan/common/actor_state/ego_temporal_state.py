from __future__ import annotations

from typing import List, Optional

from nuplan.common.actor_state.agent_temporal_state import AgentTemporalState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class EgoTemporalState(AgentTemporalState):
    """
    Temporal ego state, with future and past trajectory
    """

    def __init__(
        self,
        current_state: EgoState,
        past_trajectory: Optional[PredictedTrajectory] = None,
        predictions: Optional[List[PredictedTrajectory]] = None,
    ):
        """
        Initialize temporal state
        :param current_state: current state of ego
        :param past_trajectory: past trajectory, where last waypoint represents the same position as current state
        :param predictions: multimodal predictions, or future trajectory
        """
        super().__init__(
            initial_time_stamp=current_state.time_point, predictions=predictions, past_trajectory=past_trajectory
        )
        self._ego_current_state = current_state

    @property
    def ego_current_state(self) -> EgoState:
        """
        :return: the current ego state
        """
        return self._ego_current_state
