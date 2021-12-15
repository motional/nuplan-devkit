from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2


def _convert_absolute_to_relative_states(
    origin_absolute_state: StateSE2, absolute_states: List[StateSE2],
) -> List[StateSE2]:
    """
    Computes the relative states from a list of absolute states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param absolute_states: list of absolute poses.
    :return: list of relative states.
    """
    origin_absolute_transform = origin_absolute_state.as_matrix()
    origin_transform = np.linalg.inv(origin_absolute_transform)

    absolute_transforms = np.array([state.as_matrix() for state in absolute_states])
    relative_transforms = origin_transform @ absolute_transforms.reshape(-1, 3, 3)

    relative_states = [StateSE2.from_matrix(transform) for transform in relative_transforms]

    return relative_states


def convert_absolute_to_relative_poses(origin_absolute_state: StateSE2,
                                       absolute_states: List[StateSE2]) -> npt.NDArray[np.float32]:
    """
    Computes the relative poses from a list of absolute states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param absolute_states: list of absolute poses.
    :return: list of relative poses as numpy array.
    """
    relative_states = _convert_absolute_to_relative_states(origin_absolute_state, absolute_states)
    relative_poses = np.asarray([state.serialize() for state in relative_states])
    relative_poses = relative_poses.astype(np.float32)

    return relative_poses


def convert_absolute_to_relative_velocities(origin_absolute_velocity: StateSE2,
                                            absolute_velocities: List[StateSE2]) -> npt.NDArray[np.float32]:
    """
    Computes the relative velocities from a list of absolute velocities using an origin (anchor) velocity.

    :param origin_absolute_velocity: absolute velocities to be used as origin.
    :param absolute_velocities: list of absolute velocities.
    :return: list of relative velocities as numpy array.
    """
    relative_states = _convert_absolute_to_relative_states(origin_absolute_velocity, absolute_velocities)
    relative_velocities = np.asarray([[state.x, state.y] for state in relative_states])
    relative_velocities = relative_velocities.astype(np.float32)

    return relative_velocities
