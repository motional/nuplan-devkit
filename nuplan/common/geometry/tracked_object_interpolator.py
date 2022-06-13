from typing import List, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


def _validate_waypoints(waypoints: List[Waypoint]) -> None:
    """
    Make sure that waypoints are valid for interpolation
        raise in case they are empty or they are not monotonically increasing
    :param waypoints: list of waypoints to be interpolated
    """
    if not waypoints:
        raise RuntimeError("There are no waypoints!")

    if not np.all(np.diff([w.time_us for w in waypoints]) > 0):
        raise ValueError(f"The waypoints are not monotonically increasing: {[w.time_us for w in waypoints]}!")


def _compute_desired_time_steps(
    start_timestamp: int, end_timestamp: int, horizon_len_s: float, interval_s: float
) -> Tuple[npt.NDArray[np.float64], int]:
    """
    Compute the desired sampling
    :param start_timestamp: [us] starting time stamp
    :param end_timestamp: [us] ending time stamp
    :param horizon_len_s: [s] length of horizon
    :param interval_s: [s] interval between states
    :return: array of time stamps, and the desired length
    """
    # Extract desired time stamps
    num_future_boxes = int(horizon_len_s / interval_s)
    num_target_timestamps = num_future_boxes + 1  # include box at current frame t0
    return np.linspace(start=start_timestamp, stop=end_timestamp, num=num_target_timestamps), num_target_timestamps


def _interpolate_waypoints(
    waypoints: List[Waypoint], target_timestamps: npt.NDArray[np.float64]
) -> List[Optional[Waypoint]]:
    """
    Interpolate waypoints when required from target_timestamps
    :param waypoints: to be interpolated
    :param target_timestamps: desired sampling
    :return: list of existent interpolations, if an interpolation is not possible, it will be replaced with None
    """
    # Interpolate trajectory
    trajectory = InterpolatedTrajectory(waypoints)
    return [
        trajectory.get_state_at_time(TimePoint(t)) if trajectory.is_in_range(TimePoint(t)) else None
        for t in target_timestamps
    ]


def interpolate_future_waypoints(
    waypoints: List[Waypoint], horizon_len_s: float, interval_s: float
) -> List[Optional[Waypoint]]:
    """
    Interpolate waypoints which are in the future. If not enough waypoints are provided, we append None
    :param waypoints: list of waypoints, there needs to be at least one
    :param horizon_len_s: [s] time distance to future
    :param interval_s: [s] interval between two states
    :return: interpolated waypoints
    """
    _validate_waypoints(waypoints)

    # Extract desired time stamps
    start_timestamp = waypoints[0].time_us
    end_timestamp = int(start_timestamp + horizon_len_s * 1e6)
    target_timestamps, num_future_boxes = _compute_desired_time_steps(
        start_timestamp, end_timestamp, horizon_len_s=horizon_len_s, interval_s=interval_s
    )

    if len(waypoints) == 1:
        # Do not interpolate if trajectory is too short, and just append None
        return waypoints + [None] * (num_future_boxes - 1)

    # Interpolate trajectory
    return _interpolate_waypoints(waypoints, target_timestamps)


def interpolate_past_waypoints(
    waypoints: List[Waypoint], horizon_len_s: float, interval_s: float
) -> List[Optional[Waypoint]]:
    """
    Interpolate waypoints which are in the past. We assume that they are still monotonically increasing.
        If not enough waypoints are provided, we append None
    :param waypoints: list of waypoints, there needs to be at least one
    :param horizon_len_s: [s] time distance to past
    :param interval_s: [s] interval between two states
    :return: interpolated waypoints
    """
    _validate_waypoints(waypoints)

    # Extract desired time stamps
    end_timestamp = waypoints[-1].time_us
    start_timestamp = int(end_timestamp - horizon_len_s * 1e6)
    target_timestamps, num_future_boxes = _compute_desired_time_steps(
        start_timestamp, end_timestamp, horizon_len_s=horizon_len_s, interval_s=interval_s
    )

    if len(waypoints) == 1:
        # Do not interpolate if trajectory is too short, and just append None
        return [None] * (num_future_boxes - 1) + waypoints

    # Interpolate trajectory
    sampled_trajectory = _interpolate_waypoints(waypoints, target_timestamps)
    # Last state must exist!
    if not sampled_trajectory[-1]:
        raise RuntimeError("Last state of the trajectory has to be existent!")
    return sampled_trajectory


def interpolate_agent(agent: Agent, horizon_len_s: float, interval_s: float) -> Agent:
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
