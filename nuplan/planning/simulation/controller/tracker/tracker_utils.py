from typing import Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.geometry.compute import principal_value

DoubleMatrix = npt.NDArray[np.float64]


def _generate_profile_from_initial_condition_and_derivatives(
    initial_condition: float, derivatives: DoubleMatrix, discretization_time: float
) -> DoubleMatrix:
    """
    Returns the corresponding profile (i.e. trajectory) given an initial condition and derivatives at
    multiple timesteps by integration.
    :param initial_condition: The value of the variable at the initial timestep.
    :param derivatives: The trajectory of time derivatives of the variable at timesteps 0,..., N-1.
    :param discretization_time: [s] Time discretization used for integration.
    :return: The trajectory of the variable at timesteps 0,..., N.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."

    profile = initial_condition + np.insert(np.cumsum(derivatives * discretization_time), 0, 0.0)

    return profile  # type: ignore


def _get_position_heading_displacements_from_poses(poses: DoubleMatrix) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Returns position and heading displacements given a pose trajectory.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of poses (x, y, heading).
    :return: A tuple of position displacements and heading displacements.
    """
    assert len(poses.shape) == 2, "Expect a 2D matrix representing a trajectory of poses."
    assert poses.shape[0] > 1, "Cannot get displacements given an empty or single element pose trajectory."
    assert poses.shape[1] == 3, "Expect pose to have three elements (x, y, heading)."

    # Compute linear/angular displacements that are used to complete the kinematic state and input.
    pose_differences = np.diff(poses, axis=0)
    position_displacements = np.linalg.norm(pose_differences[:, :2], axis=1)
    heading_displacements = principal_value(pose_differences[:, 2])

    return position_displacements, heading_displacements


def _make_banded_difference_matrix(number_rows: int) -> DoubleMatrix:
    """
    Returns a banded difference matrix with specified number_rows.
    When applied to a vector [x_1, ..., x_N], it returns [x_2 - x_1, ..., x_N - x_{N-1}].
    :param number_rows: The row dimension of the banded difference matrix.
    :return: A banded difference matrix with shape (number_rows, number_rows+1).
    """
    banded_matrix: DoubleMatrix = -1.0 * np.eye(number_rows + 1, dtype=np.float64)[:-1, :]
    for ind in range(len(banded_matrix)):
        banded_matrix[ind, ind + 1] = 1.0

    return banded_matrix


def _convert_curvature_profile_to_steering_profile(
    curvature_profile: DoubleMatrix,
    discretization_time: float,
    wheel_base: float,
) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Converts from a curvature profile to the corresponding steering profile.
    We assume a kinematic bicycle model where curvature = tan(steering_angle) / wheel_base.
    For simplicity, we just use finite differences to determine steering rate.
    :param curvature_profile: [rad] Curvature trajectory to convert.
    :param discretization_time: [s] Time discretization used for integration.
    :param wheel_base: [m] The vehicle wheelbase parameter required for conversion.
    :return: The [rad] steering angle and [rad/s] steering rate (derivative) profiles.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert wheel_base > 0.0, "The vehicle's wheelbase length must be positive."

    steering_angle_profile = np.arctan(wheel_base * curvature_profile)
    steering_rate_profile = np.diff(steering_angle_profile) / discretization_time

    return steering_angle_profile, steering_rate_profile


def _fit_initial_velocity_and_acceleration_profile(
    position_displacements: DoubleMatrix, discretization_time: float, jerk_penalty: float
) -> DoubleMatrix:
    """
    Estimates initial velocity (v_0) and acceleration ({a_0, ...}) using least squares with jerk penalty regularization.
    Derivation here: https://confluence.ci.motional.com/confluence/x/huCdCg
    :param position_displacements: [m] Deviations (norm of position differences) occurring between timesteps.
    :param discretization_time: [s] Time discretization used for integration.
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :return: Least squares solution for x = [v_0, a_0, ..., a_{M-1}], given M displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert jerk_penalty > 0, "Should have a positive jerk_penalty."

    # Core problem: minimize_x ||y-Ax||_2
    y = position_displacements
    A: DoubleMatrix = np.tri(len(y), dtype=np.float64)  # lower triangular matrix
    A *= discretization_time**2
    A[:, 0] = discretization_time

    # Regularization using jerk penalty, i.e. difference of acceleration values.
    banded_matrix = _make_banded_difference_matrix(len(y) - 2)
    R: DoubleMatrix = np.block([np.zeros((len(banded_matrix), 1)), banded_matrix])

    # Compute regularized least squares solution.
    x = np.linalg.pinv(A.T @ A + jerk_penalty * R.T @ R) @ A.T @ y

    return x


def _fit_initial_curvature_and_curvature_rate_profile(
    heading_displacements: DoubleMatrix,
    velocity_profile: DoubleMatrix,
    discretization_time: float,
    curvature_rate_penalty: float,
) -> DoubleMatrix:
    """
    Estimates initial curvature (curvature_0) and curvature rate ({curvature_rate_0, ...})
    using least squares with curvature rate regularization.
    Derivation here: https://confluence.ci.motional.com/confluence/x/huCdCg
    :param heading_displacements: [rad] Angular deviations in heading occuring between timesteps.
    :param velocity_profile: [m/s] Estimated or actual velocities at the timesteps matching displacements.
    :param discretization_time: [s] Time discretization used for integration.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :return: Least squares solution for x = [curvature_0, curvature_rate_0, ..., curvature_rate_{M-1}],
             given M heading displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert curvature_rate_penalty > 0, "Should have a positive curvature_rate_penalty."

    # Core problem: minimize_x ||y-Ax||_2
    y = heading_displacements
    A: DoubleMatrix = np.tri(len(y), dtype=np.float64)  # lower triangular matrix
    A[:, 0] = velocity_profile * discretization_time

    for idx, velocity in enumerate(velocity_profile):
        if idx == 0:
            continue
        A[idx, 1:] *= velocity * discretization_time**2

    # Regularization on curvature rate.
    Q: DoubleMatrix = np.eye(len(y))
    Q[0, 0] = 0.0

    # Compute regularized least squares solution.
    x = np.linalg.pinv(A.T @ A + curvature_rate_penalty * Q) @ A.T @ y

    return x


def compute_steering_angle_feedback(
    pose_reference: DoubleMatrix, pose_current: DoubleMatrix, lookahead_distance: float, k_lateral_error: float
) -> float:
    """
    Given pose information, determines the steering angle feedback value to address initial tracking error.
    This is based on the feedback controller developed in Section 2.2 of the following paper:
    https://ddl.stanford.edu/publications/design-feedback-feedforward-steering-controller-accurate-path-tracking-and-stability
    :param pose_reference: <np.ndarray: 3,> Contains the reference pose at the current timestep.
    :param pose_current: <np.ndarray: 3,> Contains the actual pose at the current timestep.
    :param lookahead_distance: [m] Distance ahead for which we should estimate lateral error based on a linear fit.
    :param k_lateral_error: Feedback gain for lateral error used to determine steering angle feedback.
    :return: [rad] The steering angle feedback to apply.
    """
    assert pose_reference.shape == (3,), "We expect a single reference pose."
    assert pose_current.shape == (3,), "We expect a single current pose."

    assert lookahead_distance > 0.0, "Lookahead distance should be positive."
    assert k_lateral_error > 0.0, "Feedback gain for lateral error should be positive."

    x_reference, y_reference, heading_reference = pose_reference
    x_current, y_current, heading_current = pose_current

    x_error = x_current - x_reference
    y_error = y_current - y_reference
    heading_error = principal_value(heading_current - heading_reference)

    lateral_error = -x_error * np.sin(heading_reference) + y_error * np.cos(heading_reference)

    return float(-k_lateral_error * (lateral_error + lookahead_distance * heading_error))


def complete_kinematic_state_and_inputs_from_poses(
    discretization_time: float,
    wheel_base: float,
    poses: DoubleMatrix,
    jerk_penalty: float,
    curvature_rate_penalty: float,
) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Main function for joint estimation of velocity, acceleration, steering angle, and steering rate given poses
    sampled at discretization_time and the vehicle wheelbase parameter for curvature -> steering angle conversion.
    One caveat is that we can only determine the first N-1 kinematic states and N-2 kinematic inputs given
    N-1 displacement/difference values, so we need to extrapolate to match the length of poses provided.
    This is handled by repeating the last input and extrapolating the motion model for the last state.
    :param discretization_time: [s] Time discretization used for integration.
    :param wheel_base: [m] The wheelbase length for the kinematic bicycle model being used.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of poses (x, y, heading).
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :return: kinematic_states (x, y, heading, velocity, steering_angle) and corresponding
            kinematic_inputs (acceleration, steering_rate).
    """
    position_displacements, heading_displacements = _get_position_heading_displacements_from_poses(poses)

    # Compute initial velocity + acceleration least squares solution and extract results.
    initial_velocity_and_acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
        position_displacements=position_displacements,
        discretization_time=discretization_time,
        jerk_penalty=jerk_penalty,
    )
    initial_velocity = initial_velocity_and_acceleration_profile[0]
    acceleration_profile = initial_velocity_and_acceleration_profile[1:]

    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )

    # Compute initial curvature + curvature rate least squares solution and extract results.  It relies on velocity fit.
    initial_curvature_and_curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(
        heading_displacements=heading_displacements,
        velocity_profile=velocity_profile,
        discretization_time=discretization_time,
        curvature_rate_penalty=curvature_rate_penalty,
    )
    initial_curvature = initial_curvature_and_curvature_rate_profile[0]
    curvature_rate_profile = initial_curvature_and_curvature_rate_profile[1:]

    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_curvature,
        derivatives=curvature_rate_profile,
        discretization_time=discretization_time,
    )

    # Convert to steering angle given the wheelbase parameter.  At this point, we don't need to worry about curvature.
    steering_angle_profile, steering_rate_profile = _convert_curvature_profile_to_steering_profile(
        curvature_profile=curvature_profile,
        discretization_time=discretization_time,
        wheel_base=wheel_base,
    )

    # Extend input fits with a repeated element and extrapolate state fits to match length of poses.
    # This is since we fit with N-1 displacements but still have N poses at the end to deal with.
    acceleration_profile = np.append(acceleration_profile, acceleration_profile[-1])
    steering_rate_profile = np.append(steering_rate_profile, steering_rate_profile[-1])

    velocity_profile = np.append(
        velocity_profile, velocity_profile[-1] + acceleration_profile[-1] * discretization_time
    )
    steering_angle_profile = np.append(
        steering_angle_profile, steering_angle_profile[-1] + steering_rate_profile[-1] * discretization_time
    )

    # Collect completed state and input in matrices.
    kinematic_states: DoubleMatrix = np.column_stack((poses, velocity_profile, steering_angle_profile))
    kinematic_inputs: DoubleMatrix = np.column_stack((acceleration_profile, steering_rate_profile))

    return kinematic_states, kinematic_inputs
