import unittest
from functools import partial
from typing import Dict, Tuple

import numpy as np
import numpy.testing as np_test

from nuplan.common.geometry.compute import principal_value
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    DoubleMatrix,
    _convert_curvature_profile_to_steering_profile,
    _fit_initial_curvature_and_curvature_rate_profile,
    _fit_initial_velocity_and_acceleration_profile,
    _generate_profile_from_initial_condition_and_derivatives,
    _get_xy_heading_displacements_from_poses,
    _make_banded_difference_matrix,
    complete_kinematic_state_and_inputs_from_poses,
    compute_steering_angle_feedback,
    get_interpolated_reference_trajectory_poses,
    get_velocity_curvature_profiles_with_derivatives_from_poses,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


def _make_input_profiles(key_prefix: str, magnitude: float, length: int) -> Dict[str, DoubleMatrix]:
    """
    This test helper function adds input profiles to a dictionary to enable parametrized testing of the tracker utils.
    :param key_prefix: A prefix for keys in the dictionary, e.g. "curv_rate" or "acceleration".
    :param magnitude: A maximum absolute value bound for the input profile.
    :param length: How many elements (timesteps) we should have within the input profile.
    :return: A dictionary containing multiple input profiles we can apply.
    """
    acceleration_dict: Dict[str, DoubleMatrix] = {}
    acceleration_dict[f"{key_prefix}_positive"] = magnitude * np.ones(length, dtype=np.float64)
    acceleration_dict[f"{key_prefix}_zero"] = np.zeros(length, dtype=np.float64)
    acceleration_dict[f"{key_prefix}_negative"] = -magnitude * np.ones(length, dtype=np.float64)
    acceleration_dict[f"{key_prefix}_cosine"] = magnitude * np.cos(np.arange(length, dtype=np.float64))
    return acceleration_dict


def _integrate_acceleration_and_curvature_profile(
    initial_pose: DoubleMatrix,
    initial_velocity: DoubleMatrix,
    initial_curvature: DoubleMatrix,
    acceleration_profile: DoubleMatrix,
    curvature_rate_profile: DoubleMatrix,
    discretization_time: float,
) -> Tuple[DoubleMatrix, DoubleMatrix, DoubleMatrix]:
    """
    This test helper function takes in an initial state and input profile to generate the associated state trajectory.
    We use curvature for simplicity (the relationship with steering angle is 1-1 for the achievable range).
    :param initial_pose: Initial (x, y, heading) pose state.
    :param initial_velocity: [m/s] The initial velocity state.
    :param initial_curvature: [rad] The initial curvature state.
    :param acceleration_profile: [m/s^2] The acceleration input sequence to apply.
    :param curvature_rate_profile: [rad/s] The curvature rate input to apply.
    :param discretization_time: [s] Time discretization used for integration.
    :return Pose, velocity, and curvature state trajectories after integration.
    """
    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )

    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_curvature,
        derivatives=curvature_rate_profile,
        discretization_time=discretization_time,
    )

    pose_trajectory = [initial_pose]

    for (velocity, curvature) in zip(velocity_profile, curvature_profile):
        x, y, heading = pose_trajectory[-1]
        next_pose = [
            x + velocity * np.cos(heading) * discretization_time,
            y + velocity * np.sin(heading) * discretization_time,
            principal_value(heading + velocity * curvature * discretization_time),
        ]
        pose_trajectory.append(next_pose)

    return np.array(pose_trajectory), velocity_profile, curvature_profile


class TestTrackerUtils(unittest.TestCase):
    """
    Tests tracker utils, including least squares fit of kinematic states given poses.
    Throughout, we assume a kinematic bicycle model as the base dynamics model.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        # Test time discretization.
        self.test_discretization_time = 0.2

        # Least squares penalty used for regularization in fitting velocity / curvature profiles.
        # If updated, the proximity tolerances should be correspondingly updated.
        self.least_squares_penalty = 1e-10

        # Tolerances used for checking proximity between array values and expected.  Should reflect changes in the
        # selection of least_squares_penalty, relax strictness as the penalty is increased.
        self.proximity_rtol = 1e-6
        self.proximity_atol = 1e-8

        # Velocity threshold used to determine when ego is moving and thus where curvature estimation is good.
        self.moving_velocity_threshold = 0.1  # [m/s], velocity above which we say ego is moving.

        # Apply tolerances to np_test.assert_allclose to ensure consistency among test functions.
        self.assert_allclose = partial(np_test.assert_allclose, rtol=self.proximity_rtol, atol=self.proximity_atol)

        # Wheel base parameter used for curvature <-> steering angle conversion.
        self.test_wheel_base = 3.0

        # Initial state parameters.
        self.initial_pose: DoubleMatrix = np.array([5.0, 1.0, 0.1], dtype=np.float64)
        self.initial_velocity = 3.0  # [m/s]
        self.initial_curvature = 0.0  # [rad]

        # Input parameters.
        max_acceleration = 3.0  # [m/s^2]
        max_curvature_rate = 0.05  # [rad/s]
        input_length = 10

        # Generate input profiles for acceleration and curvature_rate.
        self.input_profiles = {}
        acceleration_profile_dict = _make_input_profiles(
            key_prefix="accel", magnitude=max_acceleration, length=input_length
        )
        curvature_rate_profile_dict = _make_input_profiles(
            key_prefix="curv_rate", magnitude=max_curvature_rate, length=input_length
        )

        # Generate test trajectories given cartesian product of acceleration and curvature_rate input profiles.
        for (acceleration_profile_name, acceleration_profile) in acceleration_profile_dict.items():
            for (curvature_rate_profile_name, curvature_rate_profile) in curvature_rate_profile_dict.items():
                poses, velocities, curvatures = _integrate_acceleration_and_curvature_profile(
                    initial_pose=self.initial_pose,
                    initial_velocity=self.initial_velocity,
                    initial_curvature=self.initial_curvature,
                    acceleration_profile=acceleration_profile,
                    curvature_rate_profile=curvature_rate_profile,
                    discretization_time=self.test_discretization_time,
                )
                self.input_profiles[f"{acceleration_profile_name}_{curvature_rate_profile_name}"] = {
                    "acceleration": acceleration_profile,
                    "curvature_rate": curvature_rate_profile,
                    "poses": poses,
                    "velocity": velocities,
                    "curvature": curvatures,
                }

    def test__generate_profile_from_initial_condition_and_derivatives(self) -> None:
        """
        Check that we can correctly integrate derivative profiles.
        We use a loop here to compare against the vectorized implementation.
        """
        for input_profile in self.input_profiles.values():
            velocity_profile = [self.initial_velocity]

            for acceleration in input_profile["acceleration"]:
                velocity_profile.append(velocity_profile[-1] + acceleration * self.test_discretization_time)

            self.assert_allclose(velocity_profile, input_profile["velocity"])

            curvature_profile = [self.initial_curvature]

            for curvature_rate in input_profile["curvature_rate"]:
                curvature_profile.append(curvature_profile[-1] + curvature_rate * self.test_discretization_time)

            self.assert_allclose(curvature_profile, input_profile["curvature"])

    def test__get_xy_heading_displacements_from_poses(self) -> None:
        """Get displacements and check consistency with original pose trajectory."""
        for input_profile in self.input_profiles.values():
            poses = input_profile["poses"]

            xy_displacements, heading_displacements = _get_xy_heading_displacements_from_poses(poses)

            # Displacements should have one less entry vs. the number of poses.
            self.assertEqual(len(xy_displacements), len(poses) - 1)
            self.assertEqual(len(heading_displacements), len(poses) - 1)

            # Integration of the displacements should recover the original states.
            x_integrated = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=self.initial_pose[0],
                derivatives=xy_displacements[:, 0],
                discretization_time=1.0,
            )
            y_integrated = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=self.initial_pose[1],
                derivatives=xy_displacements[:, 1],
                discretization_time=1.0,
            )

            heading_integrated = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=self.initial_pose[2],
                derivatives=heading_displacements,
                discretization_time=1.0,
            )
            heading_integrated = principal_value(heading_integrated)

            self.assert_allclose(np.column_stack((x_integrated, y_integrated, heading_integrated)), poses)

    def test__make_banded_difference_matrix(self) -> None:
        """Test that the banded difference matrix has expected structure for different sizes."""
        for test_number_rows in [1, 5, 10]:
            banded_difference_matrix = _make_banded_difference_matrix(test_number_rows)
            self.assertEqual(banded_difference_matrix.shape, (test_number_rows, test_number_rows + 1))

            # The diagonal elements should be -1 and the superdiagonal should be +1.
            self.assert_allclose(np.diag(banded_difference_matrix, k=0), -1.0)
            self.assert_allclose(np.diag(banded_difference_matrix, k=1), 1.0)

            # Check that every other element is zero by applying a diagonal/superdiagonal removing mask.
            removal_mask = np.ones_like(banded_difference_matrix)
            for idx in range(len(removal_mask)):
                removal_mask[idx, idx : (idx + 2)] = 0.0

            banded_difference_matrix_masked = np.multiply(banded_difference_matrix, removal_mask)
            self.assert_allclose(banded_difference_matrix_masked, 0.0)

    def test__convert_curvature_profile_to_steering_profile(self) -> None:
        """Check consistency of converted steering angle/rate with curvature and pose information."""
        for input_profile in self.input_profiles.values():
            curvature_profile = input_profile["curvature"]
            velocity_profile = input_profile["velocity"]
            heading_profile = input_profile["poses"][:, 2]

            steering_angle_profile, steering_rate_profile = _convert_curvature_profile_to_steering_profile(
                curvature_profile=curvature_profile,
                discretization_time=self.test_discretization_time,
                wheel_base=self.test_wheel_base,
            )

            # Check expected sizes: steering angle and curvature should match.
            self.assertEqual(len(steering_angle_profile), len(curvature_profile))
            self.assertEqual(len(steering_rate_profile), len(curvature_profile) - 1)

            # Integrating steering rate should recover steering angle.
            steering_angle_integrated = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=steering_angle_profile[0],
                derivatives=steering_rate_profile,
                discretization_time=self.test_discretization_time,
            )
            self.assert_allclose(steering_angle_integrated, steering_angle_profile)

            # If we apply speed and steering angle to the kinematic bicycle model, we should be able to recover heading.
            yawrate_profile = velocity_profile * np.tan(steering_angle_profile) / self.test_wheel_base
            heading_integrated = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=self.initial_pose[2],
                derivatives=yawrate_profile,
                discretization_time=self.test_discretization_time,
            )
            heading_integrated = principal_value(heading_integrated)
            self.assert_allclose(heading_integrated, heading_profile)

    def test__fit_initial_velocity_and_acceleration_profile(self) -> None:
        """
        Test given noiseless data and a small jerk penalty, the least squares speed/acceleration match expected values.
        """
        for input_profile in self.input_profiles.values():
            poses = input_profile["poses"]
            xy_displacements, _ = _get_xy_heading_displacements_from_poses(poses)
            heading_profile = poses[:-1, 2]  # We exclude the last heading to match xy_displacements.

            initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
                xy_displacements=xy_displacements,
                heading_profile=heading_profile,
                discretization_time=self.test_discretization_time,
                jerk_penalty=self.least_squares_penalty,
            )

            velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=initial_velocity,
                derivatives=acceleration_profile,
                discretization_time=self.test_discretization_time,
            )

            # Velocity and acceleration least squares values should be close to expected within tolerance.
            self.assert_allclose(velocity_profile, input_profile["velocity"])
            self.assert_allclose(acceleration_profile, input_profile["acceleration"])

    def test__fit_initial_curvature_and_curvature_rate_profile(self) -> None:
        """
        Test given noiseless data and a small curvature_rate penalty, the least squares curvature/curvature rate match
        expected values.  A caveat is we exclude cases where ego is stopped and thus curvature estimation is unreliable.
        """
        for input_profile in self.input_profiles.values():
            poses = input_profile["poses"]
            velocity_profile = input_profile["velocity"]
            _, heading_displacements = _get_xy_heading_displacements_from_poses(poses)

            initial_curvature, curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(
                heading_displacements=heading_displacements,
                velocity_profile=velocity_profile,
                discretization_time=self.test_discretization_time,
                curvature_rate_penalty=self.least_squares_penalty,
            )

            curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=initial_curvature,
                derivatives=curvature_rate_profile,
                discretization_time=self.test_discretization_time,
            )

            # Curvature and curvature rate least squares values should be close to expected within tolerance
            # if we use a low curvature_rate_penalty value.  The one major caveat is if the velocity is close to 0,
            # in this case, the curvature cannot be determined uniquely.  So we mask out those cases from the check.
            moving_mask = (np.abs(velocity_profile) > self.moving_velocity_threshold).astype(np.float64)
            self.assert_allclose(moving_mask * curvature_profile, moving_mask * input_profile["curvature"])

            if np.all(moving_mask > 0.0):
                # We only check curvature rate where there is continuous motion and velocity never gets close to 0.
                # This is since the neighboring curvature_rates about the timestep where velocity ~ 0 may also have
                # error relative to the expected profile.  If ego's constantly moving, this issue should not be present.
                self.assert_allclose(curvature_rate_profile, input_profile["curvature_rate"])

    def test_compute_steering_angle_feedback(self) -> None:
        """Check that sign of the steering angle feedback makes sense for various initial tracking errors."""
        # Fixed elements for this test.
        pose_reference: DoubleMatrix = self.initial_pose
        heading_reference = pose_reference[2]
        lookahead_distance = 10.0
        k_lateral_error = 0.1

        # Check the case where there's no error.
        steering_angle_zero_lateral_error = compute_steering_angle_feedback(
            pose_reference=pose_reference,
            pose_current=pose_reference,
            lookahead_distance=lookahead_distance,
            k_lateral_error=k_lateral_error,
        )
        self.assertEqual(steering_angle_zero_lateral_error, 0.0)

        # Check cases where there is only lateral error.
        for lateral_error in [-1.0, 1.0]:
            pose_lateral_error: DoubleMatrix = pose_reference + lateral_error * np.array(
                [-np.sin(heading_reference), np.cos(heading_reference), 0.0]
            )
            steering_angle_lateral_error = compute_steering_angle_feedback(
                pose_reference=pose_reference,
                pose_current=pose_lateral_error,
                lookahead_distance=lookahead_distance,
                k_lateral_error=k_lateral_error,
            )
            self.assertEqual(-np.sign(lateral_error), np.sign(steering_angle_lateral_error))

        # Check cases where there is only heading error.
        for heading_error in [-0.05, 0.05]:
            steering_angle_heading_error = compute_steering_angle_feedback(
                pose_reference=pose_reference,
                pose_current=pose_reference + [0.0, 0.0, heading_error],
                lookahead_distance=lookahead_distance,
                k_lateral_error=k_lateral_error,
            )
            self.assertEqual(-np.sign(heading_error), np.sign(steering_angle_heading_error))

    def test_get_velocity_curvature_profiles_with_derivatives_from_poses(self) -> None:
        """
        Test the joint estimation of velocity and curvature, along with their derivatives.
        Since there is overlap with complete_kinematic_state_and_inputs_from_poses,
        we just test for one given input profile and leave the extensive testing for that function.
        """
        test_input_profile = self.input_profiles["accel_cosine_curv_rate_cosine"]
        (
            velocity_profile,
            acceleration_profile,
            curvature_profile,
            curvature_rate_profile,
        ) = get_velocity_curvature_profiles_with_derivatives_from_poses(
            discretization_time=self.test_discretization_time,
            poses=test_input_profile["poses"],
            jerk_penalty=self.least_squares_penalty,
            curvature_rate_penalty=self.least_squares_penalty,
        )
        # Check that the fit is close to the expected, given zero noise and non-zero velocity.
        self.assert_allclose(velocity_profile, test_input_profile["velocity"])
        self.assert_allclose(acceleration_profile, test_input_profile["acceleration"])
        self.assert_allclose(curvature_profile, test_input_profile["curvature"])
        self.assert_allclose(curvature_rate_profile, test_input_profile["curvature_rate"])

        # Check that the integrated values are consistent with their derivatives.
        self.assert_allclose(
            np.diff(velocity_profile) / self.test_discretization_time,
            acceleration_profile,
        )

        self.assert_allclose(
            np.diff(curvature_profile) / self.test_discretization_time,
            curvature_rate_profile,
        )

    def test_complete_kinematic_state_and_inputs_from_poses(self) -> None:
        """
        Test that the joint estimation of kinematic states and inputs are consistent with expectations.
        Since there is extrapolation involved, we only compare the non-extrapolated values.
        """
        for input_profile in self.input_profiles.values():
            poses = input_profile["poses"]
            velocity_profile = input_profile["velocity"]
            acceleration_profile = input_profile["acceleration"]
            curvature_profile = input_profile["curvature"]

            kinematic_states, kinematic_inputs = complete_kinematic_state_and_inputs_from_poses(
                discretization_time=self.test_discretization_time,
                wheel_base=self.test_wheel_base,
                poses=poses,
                jerk_penalty=self.least_squares_penalty,
                curvature_rate_penalty=self.least_squares_penalty,
            )

            velocity_fit = kinematic_states[:-1, 3]
            self.assert_allclose(velocity_fit, velocity_profile)

            acceleration_fit = kinematic_inputs[:-1, 0]
            self.assert_allclose(acceleration_fit, acceleration_profile)

            steering_angle_expected, steering_rate_expected = _convert_curvature_profile_to_steering_profile(
                curvature_profile=curvature_profile,
                discretization_time=self.test_discretization_time,
                wheel_base=self.test_wheel_base,
            )

            # Similar to the test__fit_initial_curvature_and_curvature_rate_profile method, we check if there's motion
            # prior to checking steering angle / steering rate fit relative to expected.  See that method for details.
            moving_mask = (np.abs(velocity_profile) > self.moving_velocity_threshold).astype(np.float64)
            steering_angle_fit = kinematic_states[:-1, 4]
            self.assert_allclose(moving_mask * steering_angle_fit, moving_mask * steering_angle_expected)

            if np.all(moving_mask > 0.0):
                steering_rate_fit = kinematic_inputs[:-1, 1]
                self.assert_allclose(steering_rate_fit, steering_rate_expected)

    def test_get_interpolated_reference_trajectory_poses(self) -> None:
        """
        Test that we can interpolate a trajectory with constant discretization time and extract poses.
        """
        scenario = MockAbstractScenario()
        trajectory = InterpolatedTrajectory(list(scenario.get_expert_ego_trajectory()))

        expected_num_steps = 1 + int(
            (trajectory.end_time.time_s - trajectory.start_time.time_s) / self.test_discretization_time
        )

        times_s, poses = get_interpolated_reference_trajectory_poses(trajectory, self.test_discretization_time)

        # Assert pose trajectory has expected shape.
        self.assertEqual(times_s.shape, (expected_num_steps,))
        self.assertEqual(poses.shape, (expected_num_steps, 3))

        # Assert all timestamps are within bounds.
        self.assertTrue(np.all(times_s >= trajectory.start_time.time_s))
        self.assertTrue(np.all(times_s <= trajectory.end_time.time_s))

        # Assert that the timestamp spacing matches the requested discretization time.
        self.assert_allclose(np.diff(times_s), self.test_discretization_time)


if __name__ == "__main__":
    unittest.main()
