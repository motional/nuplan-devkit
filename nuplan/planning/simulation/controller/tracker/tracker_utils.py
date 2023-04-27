from typing import Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.geometry.compute import principal_value
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

DoubleMatrix = npt.NDArray[np.float64]

# Default regularization weight for initial curvature fit.  Users shouldn't really need to modify this,
# we just want it positive and small for improved conditioning of the associated least squares problem.
INITIAL_CURVATURE_PENALTY = 1e-10


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


def _get_xy_heading_displacements_from_poses(poses: DoubleMatrix) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Returns position and heading displacements given a pose trajectory.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of poses (x, y, heading).
    :return: Tuple of xy displacements with shape (num_poses-1, 2) and heading displacements with shape (num_poses-1,).
    """
    assert len(poses.shape) == 2, "Expect a 2D matrix representing a trajectory of poses."
    assert poses.shape[0] > 1, "Cannot get displacements given an empty or single element pose trajectory."
    assert poses.shape[1] == 3, "Expect pose to have three elements (x, y, heading)."

    # Compute displacements that are used to complete the kinematic state and input.
    pose_differences = np.diff(poses, axis=0)
    xy_displacements = pose_differences[:, :2]
    heading_displacements = principal_value(pose_differences[:, 2])

    return xy_displacements, heading_displacements


def _make_banded_difference_matrix(number_rows: int) -> DoubleMatrix:
    """
    Returns a banded difference matrix with specified number_rows.
    When applied to a vector [x_1, ..., x_N], it returns [x_2 - x_1, ..., x_N - x_{N-1}].
    :param number_rows: The row dimension of the banded difference matrix (e.g. N-1 in the example above).
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
    xy_displacements: DoubleMatrix, heading_profile: DoubleMatrix, discretization_time: float, jerk_penalty: float
) -> Tuple[float, DoubleMatrix]:
    """
    Estimates initial velocity (v_0) and acceleration ({a_0, ...}) using least squares with jerk penalty regularization.
    :param xy_displacements: [m] Deviations in x and y occurring between M+1 poses, a M by 2 matrix.
    :param heading_profile: [rad] Headings associated to the starting timestamp for xy_displacements, a M-length vector.
    :param discretization_time: [s] Time discretization used for integration.
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :return: Least squares solution for initial velocity (v_0) and acceleration profile ({a_0, ..., a_M-1})
             for M displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert jerk_penalty > 0, "Should have a positive jerk_penalty."

    assert len(xy_displacements.shape) == 2, "Expect xy_displacements to be a matrix."
    assert xy_displacements.shape[1] == 2, "Expect xy_displacements to have 2 columns."

    num_displacements = len(xy_displacements)  # aka M in the docstring

    assert heading_profile.shape == (
        num_displacements,
    ), "Expect the length of heading_profile to match that of xy_displacements."

    # Core problem: minimize_x ||y-Ax||_2
    y = xy_displacements.flatten()  # Flatten to a vector, [delta x_0, delta y_0, ...]

    A: DoubleMatrix = np.zeros((2 * num_displacements, num_displacements), dtype=np.float64)
    for idx_timestep, heading in enumerate(heading_profile):
        start_row = 2 * idx_timestep  # Which row of A corresponds to x-coordinate information at timestep k.

        # Related to v_0, initial velocity - column 0.
        # We fill in rows for measurements delta x_k, delta y_k.
        A[start_row : (start_row + 2), 0] = np.array(
            [
                np.cos(heading) * discretization_time,
                np.sin(heading) * discretization_time,
            ],
            dtype=np.float64,
        )

        if idx_timestep > 0:
            # Related to {a_0, ..., a_k-1}, acceleration profile - column 1 to k.
            # We fill in rows for measurements delta x_k, delta y_k.
            A[start_row : (start_row + 2), 1 : (1 + idx_timestep)] = np.array(
                [
                    [np.cos(heading) * discretization_time**2],
                    [np.sin(heading) * discretization_time**2],
                ],
                dtype=np.float64,
            )

    # Regularization using jerk penalty, i.e. difference of acceleration values.
    # If there are M displacements, then we have M - 1 acceleration values.
    # That means we have M - 2 jerk values, thus we make a banded difference matrix of that size.
    banded_matrix = _make_banded_difference_matrix(num_displacements - 2)
    R: DoubleMatrix = np.block([np.zeros((len(banded_matrix), 1)), banded_matrix])

    # Compute regularized least squares solution.
    x = np.linalg.pinv(A.T @ A + jerk_penalty * R.T @ R) @ A.T @ y

    # Extract profile from solution.
    initial_velocity = x[0]
    acceleration_profile = x[1:]

    return initial_velocity, acceleration_profile


def _fit_initial_curvature_and_curvature_rate_profile(
    heading_displacements: DoubleMatrix,
    velocity_profile: DoubleMatrix,
    discretization_time: float,
    curvature_rate_penalty: float,
    initial_curvature_penalty: float = INITIAL_CURVATURE_PENALTY,
) -> Tuple[float, DoubleMatrix]:
    """
    Estimates initial curvature (curvature_0) and curvature rate ({curvature_rate_0, ...})
    using least squares with curvature rate regularization.
    :param heading_displacements: [rad] Angular deviations in heading occuring between timesteps.
    :param velocity_profile: [m/s] Estimated or actual velocities at the timesteps matching displacements.
    :param discretization_time: [s] Time discretization used for integration.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :param initial_curvature_penalty: A regularization parameter to handle zero initial speed.  Should be positive and small.
    :return: Least squares solution for initial curvature (curvature_0) and curvature rate profile
             (curvature_rate_0, ..., curvature_rate_{M-1}) for M heading displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert curvature_rate_penalty > 0.0, "Should have a positive curvature_rate_penalty."
    assert initial_curvature_penalty > 0.0, "Should have a positive initial_curvature_penalty."

    # Core problem: minimize_x ||y-Ax||_2
    y = heading_displacements
    A: DoubleMatrix = np.tri(len(y), dtype=np.float64)  # lower triangular matrix
    A[:, 0] = velocity_profile * discretization_time

    for idx, velocity in enumerate(velocity_profile):
        if idx == 0:
            continue
        A[idx, 1:] *= velocity * discretization_time**2

    # Regularization on curvature rate.  We add a small but nonzero weight on initial curvature too.
    # This is since the corresponding row of the A matrix might be zero if initial speed is 0, leading to singularity.
    # We guarantee that Q is positive definite such that the minimizer of the least squares problem is unique.
    Q: DoubleMatrix = curvature_rate_penalty * np.eye(len(y))
    Q[0, 0] = initial_curvature_penalty

    # Compute regularized least squares solution.
    x = np.linalg.pinv(A.T @ A + Q) @ A.T @ y

    # Extract profile from solution.
    initial_curvature = x[0]
    curvature_rate_profile = x[1:]

    return initial_curvature, curvature_rate_profile


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


def get_velocity_curvature_profiles_with_derivatives_from_poses(
    discretization_time: float,
    poses: DoubleMatrix,
    jerk_penalty: float,
    curvature_rate_penalty: float,
) -> Tuple[DoubleMatrix, DoubleMatrix, DoubleMatrix, DoubleMatrix]:
    """
    Main function for joint estimation of velocity, acceleration, curvature, and curvature rate given N poses
    sampled at discretization_time.  This is done by solving two least squares problems with the given penalty weights.
    :param discretization_time: [s] Time discretization used for integration.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of N poses (x, y, heading).
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :return: Profiles for velocity (N-1), acceleration (N-2), curvature (N-1), and curvature rate (N-2).
    """
    xy_displacements, heading_displacements = _get_xy_heading_displacements_from_poses(poses)

    # Compute initial velocity + acceleration least squares solution and extract results.
    # Note: If we have M displacements, we require the M associated heading values.
    #       Therefore, we exclude the last heading in the call below.
    initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
        xy_displacements=xy_displacements,
        heading_profile=poses[:-1, 2],
        discretization_time=discretization_time,
        jerk_penalty=jerk_penalty,
    )

    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )

    # Compute initial curvature + curvature rate least squares solution and extract results.  It relies on velocity fit.
    initial_curvature, curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(
        heading_displacements=heading_displacements,
        velocity_profile=velocity_profile,
        discretization_time=discretization_time,
        curvature_rate_penalty=curvature_rate_penalty,
    )

    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_curvature,
        derivatives=curvature_rate_profile,
        discretization_time=discretization_time,
    )

    return velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile


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
    (
        velocity_profile,
        acceleration_profile,
        curvature_profile,
        curvature_rate_profile,
    ) = get_velocity_curvature_profiles_with_derivatives_from_poses(
        discretization_time=discretization_time,
        poses=poses,
        jerk_penalty=jerk_penalty,
        curvature_rate_penalty=curvature_rate_penalty,
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


def get_interpolated_reference_trajectory_poses(
    trajectory: AbstractTrajectory,
    discretization_time: float,
) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Resamples the reference trajectory at discretization_time resolution.
    It will return N times and poses, where N is a function of the trajectory duration and the discretization time.
    :param trajectory: The full trajectory from which we perform pose interpolation.
    :param discretization_time: [s] The discretization time for resampling the trajectory.
    :return An array of times in seconds (N) and an array of associated poses (N,3), sampled at the discretization time.
    """
    start_time_point = trajectory.start_time
    end_time_point = trajectory.end_time

    delta_time_point = TimePoint(int(discretization_time * 1e6))

    interpolation_times_us = np.arange(start_time_point.time_us, end_time_point.time_us, delta_time_point.time_us)

    # Adds extra state if it aligns with discretization time
    if interpolation_times_us[-1] + delta_time_point.time_us <= end_time_point.time_us:
        interpolation_times_us = np.append(
            interpolation_times_us, interpolation_times_us[-1] + delta_time_point.time_us
        )

    interpolation_time_points = [TimePoint(t_us) for t_us in interpolation_times_us]

    states = trajectory.get_state_at_times(interpolation_time_points)

    poses_interp = [[*state.rear_axle] for state in states]

    return interpolation_times_us / 1e6, np.array(poses_interp)
