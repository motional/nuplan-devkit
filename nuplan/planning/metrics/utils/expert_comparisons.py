from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.compute import principal_value
from nuplan.planning.metrics.utils.state_extractors import calculate_ego_progress_to_goal


def compute_traj_heading_errors(
    ego_traj: List[StateSE2],
    expert_traj: List[StateSE2],
) -> npt.NDArray:  # type:ignore
    """
    Compute the heading (yaw) errors between the ego trajectory and expert trajectory
    :param ego_traj: a list of StateSE2 that describe ego position with yaw
    :param expert_traj: a list of StateSE2 that describe expert position with yaw
    :return An array of yaw errors.
    """
    yaw_displacements: npt.NDArray[np.float32] = np.array(
        [ego_traj[i].heading - expert_traj[i].heading for i in range(len(ego_traj))]
    )
    heading_errors = np.abs(principal_value(yaw_displacements))

    return heading_errors  # type:ignore


def compute_traj_errors(
    ego_traj: Union[List[Point2D], List[StateSE2]],
    expert_traj: Union[List[Point2D], List[StateSE2]],
    discount_factor: float = 1.0,
    heading_diff_weight: float = 1.0,
) -> npt.NDArray:  # type:ignore
    """
    Compute the errors between the position/position_with_yaw of ego trajectory and expert trajectory
    :param ego_traj: a list of Point2D or StateSE2 that describe ego position/position with yaw
    :param expert_traj: a list of Point2D or StateSE2 that describe expert position/position with yaw
    :param discount_factor: Displacements corresponding to the k^th timestep will
    be discounted by a factor of discount_factor^k., defaults to 1.0
    :param heading_diff_weight: factor to weight heading differences if yaw errors are also
    considered, defaults to 1.0
    :return an array of displacement errors.
    """
    traj_len = len(ego_traj)
    expert_traj_len = len(expert_traj)
    assert traj_len != 0, "ego_traj should be a nonempty list"
    assert (
        traj_len == expert_traj_len or traj_len == expert_traj_len - 1
    ), "ego and expert have different trajectory lengths"

    # Compute the differences
    displacements = np.zeros((traj_len, 2))
    for i in range(traj_len):
        displacements[i, :] = [ego_traj[i].x - expert_traj[i].x, ego_traj[i].y - expert_traj[i].y]

    dist_seq = np.hypot(displacements[:, 0], displacements[:, 1])

    if isinstance(ego_traj[0], StateSE2) and isinstance(expert_traj[0], StateSE2) and heading_diff_weight != 0:
        heading_errors = compute_traj_heading_errors(ego_traj, expert_traj)
        weighted_heading_errors = heading_errors * heading_diff_weight
        dist_seq = dist_seq + weighted_heading_errors

    # Discount the errors in time
    if discount_factor != 1:
        discount_weights = get_discount_weights(discount_factor=discount_factor, traj_len=traj_len)
        dist_seq = np.multiply(dist_seq, discount_weights)

    return dist_seq  # type:ignore


def get_discount_weights(
    discount_factor: float, traj_len: int, num_trajs: int = 1
) -> Optional[npt.NDArray[np.float32]]:
    """
    Return the trajectory discount weight array if applicable
    :param discount_factor: the discount factor by which the displacements corresponding to the k^th timestep will
    be discounted
    :param traj_len: len of traj
    :param optional num_trajs: num of ego trajs, default is set to 1, but it's generalized in case we need to
    compare multiple ego trajs with expert
    :return array of discount_weights.
    """
    discount_weights = None
    if discount_factor != 1.0:
        # Compute discount_factors
        pow_arr = np.tile(np.arange(traj_len), (num_trajs, 1))  # type:ignore
        discount_weights = np.power(discount_factor, pow_arr)
    return discount_weights


def calculate_relative_progress_to_goal(
    ego_states: List[EgoState], expert_states: List[EgoState], goal: StateSE2, tolerance: float = 0.1
) -> float:
    """
    Ratio of ego's to the expert's progress towards goal rounded up
    :param ego_states: A list of ego states
    :param expert_states: A list of expert states
    :param goal: goal
    :param tolerance: tolerance used for round up
    :return Ratio of progress towards goal.
    """
    ego_progress_value = calculate_ego_progress_to_goal(ego_states, goal)
    expert_progress_value = calculate_ego_progress_to_goal(expert_states, goal)
    relative_progress: float = max(tolerance, ego_progress_value) / max(tolerance, expert_progress_value)

    return relative_progress
