from typing import List

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateSE2


def _convert_absolute_to_relative_states(
    origin_absolute_state: StateSE2,
    absolute_states: List[StateSE2],
) -> List[StateSE2]:
    """
    Computes the relative states from a list of absolute states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param absolute_states: list of absolute poses.
    :return: list of relative states.
    """
    origin_absolute_transform = origin_absolute_state.as_matrix()
    origin_transform = np.linalg.inv(origin_absolute_transform)

    absolute_transforms: npt.NDArray[np.float32] = np.array([state.as_matrix() for state in absolute_states])
    relative_transforms = origin_transform @ absolute_transforms.reshape(-1, 3, 3)

    relative_states = [StateSE2.from_matrix(transform) for transform in relative_transforms]

    return relative_states


def _convert_relative_to_absolute_states(
    origin_absolute_state: StateSE2,
    relative_states: List[StateSE2],
) -> List[StateSE2]:
    """
    Computes the absolute states from a list of relative states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param relative_states: list of relative poses.
    :return: list of absolute states.
    """
    origin_transform = origin_absolute_state.as_matrix()

    relative_transforms: npt.NDArray[np.float32] = np.array([state.as_matrix() for state in relative_states])
    absolute_transforms = origin_transform @ relative_transforms.reshape(-1, 3, 3)

    absolute_states = [StateSE2.from_matrix(transform) for transform in absolute_transforms]

    return absolute_states


def convert_absolute_to_relative_poses(
    origin_absolute_state: StateSE2, absolute_states: List[StateSE2]
) -> npt.NDArray[np.float32]:
    """
    Computes the relative poses from a list of absolute states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param absolute_states: list of absolute poses.
    :return: list of relative poses as numpy array.
    """
    relative_states = _convert_absolute_to_relative_states(origin_absolute_state, absolute_states)
    relative_poses: npt.NDArray[np.float32] = np.asarray([state.serialize() for state in relative_states]).astype(
        np.float32
    )

    return relative_poses


def convert_relative_to_absolute_poses(
    origin_absolute_state: StateSE2, relative_states: List[StateSE2]
) -> npt.NDArray[np.float64]:
    """
    Computes the absolute poses from a list of relative states using an origin (anchor) state.

    :param origin_absolute_state: absolute state to be used as origin.
    :param relative_states: list of absolute poses.
    :return: list of relative poses as numpy array.
    """
    absolute_states = _convert_relative_to_absolute_states(origin_absolute_state, relative_states)
    absolute_poses: npt.NDArray[np.float64] = np.asarray([state.serialize() for state in absolute_states]).astype(
        np.float64
    )

    return absolute_poses


def convert_absolute_to_relative_velocities(
    origin_absolute_velocity: StateSE2, absolute_velocities: List[StateSE2]
) -> npt.NDArray[np.float32]:
    """
    Computes the relative velocities from a list of absolute velocities using an origin (anchor) velocity.

    :param origin_absolute_velocity: absolute velocities to be used as origin.
    :param absolute_velocities: list of absolute velocities.
    :return: list of relative velocities as numpy array.
    """
    relative_states = _convert_absolute_to_relative_states(origin_absolute_velocity, absolute_velocities)
    relative_velocities: npt.NDArray[np.float32] = np.asarray([[state.x, state.y] for state in relative_states]).astype(
        np.float32
    )

    return relative_velocities
