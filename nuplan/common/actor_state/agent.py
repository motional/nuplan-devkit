from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint


class AgentType(Enum):
    UNKNOWN = 'unknown'
    VEHICLE = 'vehicle'
    PEDESTRIAN = 'pedestrian'
    BICYCLE = 'bicycle'
    GENERIC_OBJECT = 'generic_object'
    TRAFFIC_CONE = 'traffic_cone'
    BARRIER = 'barrier'
    CZONE_SIGN = 'czone_sign'
    EGO = 'ego'

    @classmethod
    def _missing_(cls, _: object) -> AgentType:
        """ Assigns the UNKNOWN type to missing labels. """
        return AgentType.UNKNOWN


@dataclass
class Waypoint:
    """ Represents a waypoint which is part of a trajectory. Optionals to allow for geometric trajectory"""
    future_time: Optional[TimePoint]  # TimePoint of this Waypoint, in global time
    oriented_box: OrientedBox  # Contains pose of the agent, and lazy evaluated Polygon
    velocity: Optional[StateVector2D]  # Predicted velocity of the agent at the waypoint


@dataclass
class PredictedTrajectory:
    """ Stores a predicted trajectory, along with its probability. """
    probability: float  # Probability assigned to this trajectory prediction
    waypoints: List[Waypoint]  # Waypoints of the predicted trajectory, the first waypoint is the current state of ego


class Agent:
    def __init__(self, token: str, agent_type: AgentType, oriented_box: OrientedBox, velocity: StateVector2D,
                 angular_velocity: Optional[float] = None, predictions: Optional[List[PredictedTrajectory]] = None,
                 track_token: Optional[str] = None):
        """
        Representation of an Agent in the scene.
        :param token: Unique token.
        :param agent_type: Type of the current agent
        :param oriented_box: Geometrical representation of the Agent
        :param velocity: Velocity (vectorial) of Agent
        :param angular_velocity: The scalar angular velocity of the agent, if available
        :param predictions: Optional list of (possibly multiple) predicted trajectories
        :param track_token: Track token in the "track" table that corresponds to a particular box.
        """
        self._token = token
        self._track_token = track_token
        self._agent_type = agent_type

        self._box: OrientedBox = oriented_box
        self._velocity = velocity
        self._angular_velocity = angular_velocity

        # Possible multiple predicted trajectories
        self._predictions: List[PredictedTrajectory] = predictions if predictions is not None else []

    @property
    def token(self) -> str:
        return self._token

    @property
    def track_token(self) -> Optional[str]:
        return self._track_token

    @property
    def agent_type(self) -> AgentType:
        return self._agent_type

    @property
    def box(self) -> OrientedBox:
        return self._box

    @property
    def velocity(self) -> StateVector2D:
        return self._velocity

    @property
    def angular_velocity(self) -> Optional[float]:
        return self._angular_velocity

    @property
    def predictions(self) -> List[PredictedTrajectory]:
        return self._predictions

    @predictions.setter
    def predictions(self, predicted_trajectories: List[PredictedTrajectory]) -> None:
        """ Setter for predicted trajectories, checks if the listed probabilities sum to one. """
        # Sanity check that if predictions are provided, probabilities sum to 1
        probability_sum = sum(prediction.probability for prediction in predicted_trajectories)
        if not abs(probability_sum - 1) < 1e-6 and predicted_trajectories:
            raise ValueError(f"The provided trajectory probabilities did not sum to one, but to {probability_sum:.2f}!")
        self._predictions = predicted_trajectories
