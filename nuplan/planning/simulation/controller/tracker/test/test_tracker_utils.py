import unittest

import numpy as np
import numpy.testing as np_test

from nuplan.common.geometry.compute import principal_value
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    DoubleMatrix,
    _convert_curvature_profile_to_steering_profile,
    _fit_initial_curvature_and_curvature_rate_profile,
    _fit_initial_velocity_and_acceleration_profile,
    _generate_profile_from_initial_condition_and_derivatives,
    _get_position_heading_displacements_from_poses,
    _make_banded_difference_matrix,
    complete_kinematic_state_and_inputs_from_poses,
    compute_steering_angle_feedback,
)


class TestTrackerUtils(unittest.TestCase):
    """
    Tests tracker utils, including least squares fit of kinematic states given poses.
    Throughout, we assume a kinematic bicycle model as the base dynamics model.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        # Test time discretization matching below data.  If modified, need to updated trajectory information below.
        self.test_discretization_time = 0.2

        # Absolute tolerance used for checking proximity between array values and expected.
        # Used to determine if least squares solutions are close to expected values.
        self.least_squares_proximity_atol = 1e-3

        # Relative tolerance for np.allclose to pass.  Used when there should be a perfect match, not for least squares.
        self.proximity_rtol = 1e-6

        # Wheel base parameter used for curvature <-> steering angle conversion.
        self.test_wheel_base = 3.0

        # Poses (x, y, heading) generated using integration of below kinematic state/input values
        # with self.test_discretization_time.
        self.test_poses: DoubleMatrix = np.array(
            [
                [5.0, 1.0, 0.1],
                [5.99500417, 1.09983342, 0.1],
                [7.06960866, 1.20765351, 0.1108],
                [8.2209106, 1.33574236, 0.1337372],
                [9.44199886, 1.5000279, 0.16948171],
                [10.72151691, 1.71898325, 0.21785556],
                [12.04336971, 2.01160025, 0.2777388],
                [13.3869085, 2.39465357, 0.34708237],
                [14.72793736, 2.87973805, 0.42303224],
                [16.04069235, 3.4707593, 0.50215324],
                [17.30057946, 4.16256541, 0.58072617],
                [18.48708172, 4.94115628, 0.65508114],
            ],
            dtype=np.float64,
        )

        # Velocity profile matching test_poses.
        self.velocity_expected: DoubleMatrix = np.array(
            [
                5.0,
                5.4,
                5.79202663,
                6.16045103,
                6.49058527,
                6.76926796,
                6.98538888,
                7.13033198,
                7.19831884,
                7.18663903,
                7.09575819,
            ],
            dtype=np.float64,
        )

        # Curvature profile matching test_poses.
        self.curvature_expected: DoubleMatrix = np.array(
            [
                0.0,
                0.01,
                0.01980067,
                0.02901128,
                0.03726463,
                0.0442317,
                0.04963472,
                0.0532583,
                0.05495797,
                0.05466598,
                0.05239395,
            ],
            dtype=np.float64,
        )

        # Acceleration profile matching test_poses.
        self.acceleration_expected: DoubleMatrix = np.array(
            [
                2.0,
                1.96013316,
                1.84212199,
                1.65067123,
                1.39341342,
                1.08060461,
                0.72471551,
                0.33993429,
                -0.05839904,
                -0.45440419,
            ],
            dtype=np.float64,
        )

        # Curvature rate profile matching test_poses.
        self.curvature_rate_expected: DoubleMatrix = np.array(
            [
                0.05,
                0.04900333,
                0.04605305,
                0.04126678,
                0.03483534,
                0.02701512,
                0.01811789,
                0.00849836,
                -0.00145998,
                -0.0113601,
            ],
            dtype=np.float64,
        )

    def test__generate_profile_from_initial_condition_and_derivatives(self) -> None:
        """Check that we can correctly integrate derivative profiles."""
        velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=self.velocity_expected[0],
            derivatives=self.acceleration_expected,
            discretization_time=self.test_discretization_time,
        )
        np_test.assert_allclose(velocity_profile, self.velocity_expected, rtol=self.proximity_rtol)

        curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=self.curvature_expected[0],
            derivatives=self.curvature_rate_expected,
            discretization_time=self.test_discretization_time,
        )
        np_test.assert_allclose(curvature_profile, self.curvature_expected, rtol=self.proximity_rtol)

    def test__get_position_heading_displacements_from_poses(self) -> None:
        """Get displacements and check consistency with original pose trajectory."""
        position_displacements, heading_displacements = _get_position_heading_displacements_from_poses(self.test_poses)

        # Displacements should have one less entry vs. the number of poses.
        self.assertEqual(len(position_displacements), len(self.test_poses) - 1)
        self.assertEqual(len(heading_displacements), len(self.test_poses) - 1)

        # Position displacements should correspond to expected velocity due to noiseless inputs.
        np_test.assert_allclose(
            position_displacements / self.test_discretization_time, self.velocity_expected, rtol=self.proximity_rtol
        )

        # The displacement information should also match expected curvature for the same reason.
        np_test.assert_allclose(
            heading_displacements / position_displacements, self.curvature_expected, rtol=self.proximity_rtol
        )

    def test__make_banded_difference_matrix(self) -> None:
        """Test that the banded difference matrix has expected structure for different sizes."""
        for test_number_rows in [1, 5, 10]:
            banded_difference_matrix = _make_banded_difference_matrix(test_number_rows)
            self.assertEqual(banded_difference_matrix.shape, (test_number_rows, test_number_rows + 1))

            # The diagonal elements should be -1 and the superdiagonal should be +1.
            np_test.assert_allclose(np.diag(banded_difference_matrix, k=0), -1.0, rtol=self.proximity_rtol)
            np_test.assert_allclose(np.diag(banded_difference_matrix, k=1), 1.0, rtol=self.proximity_rtol)

            # Check that every other element is zero by applying a diagonal/superdiagonal removing mask.
            removal_mask = np.ones_like(banded_difference_matrix)
            for idx in range(len(removal_mask)):
                removal_mask[idx, idx : (idx + 2)] = 0.0

            banded_difference_matrix_masked = np.multiply(banded_difference_matrix, removal_mask)
            np_test.assert_allclose(banded_difference_matrix_masked, 0.0, rtol=self.proximity_rtol)

    def test__convert_curvature_profile_to_steering_profile(self) -> None:
        """Check consistency of converted steering angle/rate with curvature and pose information."""
        steering_angle_profile, steering_rate_profile = _convert_curvature_profile_to_steering_profile(
            curvature_profile=self.curvature_expected,
            discretization_time=self.test_discretization_time,
            wheel_base=self.test_wheel_base,
        )

        # Check expected sizes: steering angle and curvature should match.
        self.assertEqual(len(steering_angle_profile), len(self.curvature_expected))
        self.assertEqual(len(steering_rate_profile), len(self.curvature_expected) - 1)

        # Integrating steering rate should recover steering angle.
        steering_angle_refit = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=steering_angle_profile[0],
            derivatives=steering_rate_profile,
            discretization_time=self.test_discretization_time,
        )
        np_test.assert_allclose(steering_angle_refit, steering_angle_profile, rtol=self.proximity_rtol)

        # If we apply speed and steering angle to the kinematic bicycle model,
        # we should be able to recover the heading profile.
        yawrate_profile = self.velocity_expected * np.tan(steering_angle_profile) / self.test_wheel_base
        heading_profile = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=self.test_poses[0, 2],
            derivatives=yawrate_profile,
            discretization_time=self.test_discretization_time,
        )
        np_test.assert_allclose(heading_profile, self.test_poses[:, 2], rtol=self.proximity_rtol)

    def test__fit_initial_velocity_and_acceleration_profile(self) -> None:
        """
        Test given noiseless data and a small jerk penalty, the least squares speed and acceleration
        match expected values.  If the jerk penalty is very large, we expect a less accurate fit.
        """
        position_displacements = np.linalg.norm(np.diff(self.test_poses[:, :2], axis=0), axis=1)

        for (jerk_penalty, expect_close) in zip([1e-10, 100.0], [True, False]):
            initial_velocity_and_acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
                position_displacements=position_displacements,
                discretization_time=self.test_discretization_time,
                jerk_penalty=jerk_penalty,
            )

            initial_velocity = initial_velocity_and_acceleration_profile[0]
            acceleration_profile = initial_velocity_and_acceleration_profile[1:]

            velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=initial_velocity,
                derivatives=acceleration_profile,
                discretization_time=self.test_discretization_time,
            )

            # Velocity and acceleration least squares values should be close to expected within tolerance
            # if we use a low jerk_penalty value.
            self.assertEqual(
                expect_close,
                np.allclose(velocity_profile, self.velocity_expected, atol=self.least_squares_proximity_atol),
            )
            self.assertEqual(
                expect_close,
                np.allclose(acceleration_profile, self.acceleration_expected, atol=self.least_squares_proximity_atol),
            )

    def test__fit_initial_curvature_and_curvature_rate_profile(self) -> None:
        """
        Test given noiseless data and a small curvature_rate penalty, the least squares curvature and curvature rate
        match expected values.  If the curvature rate penalty is very large, we expect a less accurate fit.
        """
        heading_displacements = principal_value(np.diff(self.test_poses[:, 2]))

        for (curvature_rate_penalty, expect_close) in zip([1e-10, 100.0], [True, False]):
            initial_curvature_and_curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(
                heading_displacements=heading_displacements,
                velocity_profile=self.velocity_expected,
                discretization_time=self.test_discretization_time,
                curvature_rate_penalty=curvature_rate_penalty,
            )

            initial_curvature = initial_curvature_and_curvature_rate_profile[0]
            curvature_rate_profile = initial_curvature_and_curvature_rate_profile[1:]

            curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=initial_curvature,
                derivatives=curvature_rate_profile,
                discretization_time=self.test_discretization_time,
            )

            # Curvature and curvature rate least squares values should be close to expected within tolerance
            # if we use a low curvature_rate_penalty value.
            self.assertEqual(
                expect_close,
                np.allclose(curvature_profile, self.curvature_expected, atol=self.least_squares_proximity_atol),
            )
            self.assertEqual(
                expect_close,
                np.allclose(
                    curvature_rate_profile, self.curvature_rate_expected, atol=self.least_squares_proximity_atol
                ),
            )

    def test_compute_steering_angle_feedback(self) -> None:
        """Check that sign of the steering angle feedback makes sense for various initial tracking errors."""
        # Fixed elements for this test.
        pose_reference: DoubleMatrix = np.array([1.0, 5.0, 0.1], dtype=np.float64)
        heading_reference = pose_reference[2]
        lookahead_distance = 10.0
        k_lateral_error = 0.1

        # Case 1: Positive lateral error.
        pose_current_positive_lateral_error: DoubleMatrix = pose_reference + np.array(
            [-np.sin(heading_reference), np.cos(heading_reference), 0.0]
        )
        steering_angle_positive_lateral_error = compute_steering_angle_feedback(
            pose_reference=pose_reference,
            pose_current=pose_current_positive_lateral_error,
            lookahead_distance=lookahead_distance,
            k_lateral_error=k_lateral_error,
        )
        self.assertEqual(np.sign(steering_angle_positive_lateral_error), -1.0)

        # Case 2: Negative lateral error.
        pose_current_negative_lateral_error: DoubleMatrix = pose_reference - np.array(
            [-np.sin(heading_reference), np.cos(heading_reference), 0.0]
        )
        steering_angle_negative_lateral_error = compute_steering_angle_feedback(
            pose_reference=pose_reference,
            pose_current=pose_current_negative_lateral_error,
            lookahead_distance=lookahead_distance,
            k_lateral_error=k_lateral_error,
        )
        self.assertEqual(np.sign(steering_angle_negative_lateral_error), 1.0)

        # Case 3: No lateral error.  We also check the impact of heading error by itself here.
        steering_angle_zero_lateral_error = compute_steering_angle_feedback(
            pose_reference=pose_reference,
            pose_current=pose_reference,
            lookahead_distance=lookahead_distance,
            k_lateral_error=k_lateral_error,
        )
        self.assertEqual(steering_angle_zero_lateral_error, 0.0)

        for heading_error in [-0.05, 0.05]:
            steering_angle_heading_error = compute_steering_angle_feedback(
                pose_reference=pose_reference,
                pose_current=pose_reference + [0.0, 0.0, heading_error],
                lookahead_distance=lookahead_distance,
                k_lateral_error=k_lateral_error,
            )
            self.assertEqual(-np.sign(heading_error), np.sign(steering_angle_heading_error))

    def test_complete_kinematic_state_and_inputs_from_poses(self) -> None:
        """
        Test that the joint estimation of kinematic states and inputs are consistent with expectations.
        Since there is extrapolation involved, we only compare the non-extrapolated values.
        """
        kinematic_states, kinematic_inputs = complete_kinematic_state_and_inputs_from_poses(
            discretization_time=self.test_discretization_time,
            wheel_base=self.test_wheel_base,
            poses=self.test_poses,
            jerk_penalty=1e-6,
            curvature_rate_penalty=1e-6,
        )

        velocity_fit = kinematic_states[:-1, 3]
        np_test.assert_allclose(velocity_fit, self.velocity_expected, atol=self.least_squares_proximity_atol)

        acceleration_fit = kinematic_inputs[:-1, 0]

        np_test.assert_allclose(acceleration_fit, self.acceleration_expected, atol=self.least_squares_proximity_atol)

        steering_angle_expected, steering_rate_expected = _convert_curvature_profile_to_steering_profile(
            curvature_profile=self.curvature_expected,
            discretization_time=self.test_discretization_time,
            wheel_base=self.test_wheel_base,
        )

        steering_angle_fit = kinematic_states[:-1, 4]
        np_test.assert_allclose(steering_angle_fit, steering_angle_expected, atol=self.least_squares_proximity_atol)

        steering_rate_fit = kinematic_inputs[:-1, 1]
        np_test.assert_allclose(steering_rate_fit, steering_rate_expected, atol=self.least_squares_proximity_atol)


if __name__ == "__main__":
    unittest.main()
