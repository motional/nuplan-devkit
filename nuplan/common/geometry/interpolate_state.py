from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


def _validate_waypoints(waypoints: List[InterpolatableState]) -> None:
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
    waypoints: List[InterpolatableState], target_timestamps: npt.NDArray[np.float64], pad_with_none: bool = True
) -> List[Optional[InterpolatableState]]:
    """
    Interpolate waypoints when required from target_timestamps
    :param waypoints: to be interpolated
    :param target_timestamps: desired sampling
    :param pad_with_none: if True, the output will have None for states that can not be interpolated
    :return: list of existent interpolations, if an interpolation is not possible, it will be replaced with None
    """
    # Interpolate trajectory
    trajectory = InterpolatedTrajectory(waypoints)
    if pad_with_none:
        return [
            trajectory.get_state_at_time(TimePoint(t)) if trajectory.is_in_range(TimePoint(t)) else None
            for t in target_timestamps
        ]
    return [
        trajectory.get_state_at_time(TimePoint(t)) for t in target_timestamps if trajectory.is_in_range(TimePoint(t))
    ]


def interpolate_future_waypoints(
    waypoints: List[InterpolatableState], horizon_len_s: float, interval_s: float
) -> List[Optional[InterpolatableState]]:
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
        return waypoints + cast(List[Optional[InterpolatableState]], [None] * (num_future_boxes - 1))

    # Interpolate trajectory
    return _interpolate_waypoints(waypoints, target_timestamps)


def interpolate_past_waypoints(
    waypoints: List[InterpolatableState], horizon_len_s: float, interval_s: float
) -> List[Optional[InterpolatableState]]:
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
    start_timestamp = max(int(end_timestamp - horizon_len_s * 1e6), 0)
    target_timestamps, num_future_boxes = _compute_desired_time_steps(
        start_timestamp, end_timestamp, horizon_len_s=horizon_len_s, interval_s=interval_s
    )

    if len(waypoints) == 1:
        # Do not interpolate if trajectory is too short, and just append None
        return cast(List[Optional[InterpolatableState]], [None] * (num_future_boxes - 1)) + waypoints

    # Interpolate trajectory
    sampled_trajectory = _interpolate_waypoints(waypoints, target_timestamps)
    # Last state must exist!
    if not sampled_trajectory[-1]:
        raise RuntimeError("Last state of the trajectory has to be existent!")
    return sampled_trajectory
