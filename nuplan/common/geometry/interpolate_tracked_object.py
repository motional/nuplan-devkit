from typing import List, Union, cast

from nuplan.common.actor_state.agent_temporal_state import AgentTemporalState
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.geometry.interpolate_state import interpolate_future_waypoints, interpolate_past_waypoints
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


def interpolate_agent(agent: AgentTemporalState, horizon_len_s: float, interval_s: float) -> AgentTemporalState:
    """
    Interpolate agent's future predictions and past trajectory based on the predefined length and interval
    :param agent: to be interpolated
    :param horizon_len_s: [s] horizon of predictions
    :param interval_s: [s] interval between two states
    :return: interpolated agent, where missing waypoints are replaced with None
    """
    interpolated_agent = agent
    if interpolated_agent.predictions:
        interpolated_agent.predictions = [
            PredictedTrajectory(
                waypoints=interpolate_future_waypoints(
                    mode.waypoints, horizon_len_s=horizon_len_s, interval_s=interval_s
                ),
                probability=mode.probability,
            )
            for mode in interpolated_agent.predictions
        ]

    past_trajectory = interpolated_agent.past_trajectory
    if past_trajectory:
        interpolated_agent.past_trajectory = PredictedTrajectory(
            waypoints=interpolate_past_waypoints(
                past_trajectory.waypoints, horizon_len_s=horizon_len_s, interval_s=interval_s
            ),
            probability=past_trajectory.probability,
        )
    return interpolated_agent


def interpolate_tracks(
    tracked_objects: Union[TrackedObjects, List[TrackedObject]], horizon_len_s: float, interval_s: float
) -> List[TrackedObject]:
    """
    Interpolate agent's predictions and past trajectory, if not enough states are present, add NONE!
    :param tracked_objects: agents to be interpolated
    :param horizon_len_s: [s] horizon from initial waypoint
    :param interval_s: [s] interval between two states
    :return: interpolated agents
    """
    all_tracked_objects = (
        tracked_objects if isinstance(tracked_objects, TrackedObjects) else TrackedObjects(tracked_objects)
    )
    return [
        interpolate_agent(agent, horizon_len_s=horizon_len_s, interval_s=interval_s)
        for agent in all_tracked_objects.get_agents()
    ] + cast(List[TrackedObject], all_tracked_objects.get_static_objects())
