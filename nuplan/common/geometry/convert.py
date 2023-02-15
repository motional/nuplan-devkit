from typing import List

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D


def pose_from_matrix(transform_matrix: npt.NDArray[np.float32]) -> StateSE2:
    """
    Converts a 3x3 transformation matrix to a 2D pose
    :param transform_matrix: 3x3 transformation matrix
    :return: 2D pose (x, y, yaw)
    """
    if transform_matrix.shape != (3, 3):
        raise RuntimeError(f"Expected a 3x3 transformation matrix, got {transform_matrix.shape}")

    heading = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

    return StateSE2(transform_matrix[0, 2], transform_matrix[1, 2], heading)


def matrix_from_pose(pose: StateSE2) -> npt.NDArray[np.float64]:
    """
    Converts a 2D pose to a 3x3 transformation matrix

    :param pose: 2D pose (x, y, yaw)
    :return: 3x3 transformation matrix
    """
    return np.array(
        [
            [np.cos(pose.heading), -np.sin(pose.heading), pose.x],
            [np.sin(pose.heading), np.cos(pose.heading), pose.y],
            [0, 0, 1],
        ]
    )


def absolute_to_relative_poses(absolute_poses: List[StateSE2]) -> List[StateSE2]:
    """
    Converts a list of SE2 poses from absolute to relative coordinates with the first pose being the origin
    :param absolute_poses: list of absolute poses to convert
    :return: list of converted relative poses
    """
    absolute_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(pose) for pose in absolute_poses])
    origin_transform = np.linalg.inv(absolute_transforms[0])
    relative_transforms = origin_transform @ absolute_transforms
    relative_poses = [pose_from_matrix(transform_matrix) for transform_matrix in relative_transforms]

    return relative_poses


def relative_to_absolute_poses(origin_pose: StateSE2, relative_poses: List[StateSE2]) -> List[StateSE2]:
    """
    Converts a list of SE2 poses from relative to absolute coordinates using an origin pose.
    :param origin_pose: Reference origin pose
    :param relative_poses: list of relative poses to convert
    :return: list of converted absolute poses
    """
    relative_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(pose) for pose in relative_poses])
    origin_transform = matrix_from_pose(origin_pose)
    absolute_transforms: npt.NDArray[np.float32] = origin_transform @ relative_transforms
    absolute_poses = [pose_from_matrix(transform_matrix) for transform_matrix in absolute_transforms]

    return absolute_poses


def numpy_array_to_absolute_velocity(
    origin_absolute_state: StateSE2, velocities: npt.NDArray[np.float32]
) -> List[StateVector2D]:
    """
    Converts an array of relative numpy velocities to a list of absolute StateVector2D objects.
    :param velocities: list of velocities to convert
    :param origin_absolute_state: Reference origin pose
    :return: list of StateVector2D
    """
    assert velocities.shape[1] == 2, f"Expected poses shape of (*, 2), got {velocities.shape}"
    velocities = np.pad(velocities.astype(np.float64), ((0, 0), (0, 1)), "constant", constant_values=0.0)
    relative_states = [StateSE2.deserialize(pose) for pose in velocities]
    return [
        StateVector2D(state.x, state.y) for state in relative_to_absolute_poses(origin_absolute_state, relative_states)
    ]


def numpy_array_to_absolute_pose(origin_absolute_state: StateSE2, poses: npt.NDArray[np.float32]) -> List[StateSE2]:
    """
    Converts an array of relative numpy poses to a list of absolute StateSE2 objects.
    :param poses: list of poses to convert
    :param origin_absolute_state: Reference origin pose
    :return: list of StateSE2
    """
    assert poses.shape[1] == 3, f"Expected poses shape of (*, 3), got {poses.shape}"
    relative_states = [StateSE2.deserialize(pose) for pose in poses]
    return relative_to_absolute_poses(origin_absolute_state, relative_states)


def vector_2d_from_magnitude_angle(magnitude: float, angle: float) -> StateVector2D:
    """
    Projects magnitude and angle into a vector of x-y components.
    :param magnitude: The magnitude of the vector.
    :param angle: The angle of the vector.
    :return: A state vector.
    """
    return StateVector2D(np.cos(angle) * magnitude, np.sin(angle) * magnitude)
