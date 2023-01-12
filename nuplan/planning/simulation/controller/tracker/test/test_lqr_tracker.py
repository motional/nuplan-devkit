import itertools
import unittest

import numpy as np
import numpy.testing as np_test
import numpy.typing as npt

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class TestLQRTracker(unittest.TestCase):
    """
    Tests LQR Tracker.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.initial_time_point = TimePoint(0)
        self.scenario = MockAbstractScenario(initial_time_us=self.initial_time_point)
        self.trajectory = InterpolatedTrajectory(list(self.scenario.get_expert_ego_trajectory()))

        # The sampling time [s] defines the duration between the current and next iteration.
        # The discretization time used for the tracker can vary as currently implemented.
        self.sampling_time = 0.5

        self.tracker = LQRTracker(
            q_longitudinal=[10.0],
            r_longitudinal=[1.0],
            q_lateral=[1.0, 10.0, 0.0],
            r_lateral=[1.0],
            discretization_time=0.1,
            tracking_horizon=10,
            jerk_penalty=1e-4,
            curvature_rate_penalty=1e-2,
            stopping_proportional_gain=0.5,
            stopping_velocity=0.2,
        )

    def test_track_trajectory(self) -> None:
        """Ensure we are able to run track trajectory using LQR."""
        dynamic_state = self.tracker.track_trajectory(
            current_iteration=SimulationIteration(self.initial_time_point, 0),
            next_iteration=SimulationIteration(TimePoint(int(self.sampling_time * 1e6)), 1),
            initial_state=self.scenario.initial_ego_state,
            trajectory=self.trajectory,
        )

        # We check existence of the key fields needed to propagate the system based on the computed control input.
        self.assertIsInstance(dynamic_state._rear_axle_to_center_dist, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_velocity_2d.x, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_velocity_2d.y, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_acceleration_2d.x, (int, float))
        self.assertIsInstance(dynamic_state.rear_axle_acceleration_2d.y, (int, float))
        self.assertIsInstance(dynamic_state.tire_steering_rate, (int, float))

        # Some of the key fields need to satisfy constraints, which we check here.
        self.assertGreater(dynamic_state._rear_axle_to_center_dist, 0.0)
        self.assertEqual(dynamic_state.rear_axle_acceleration_2d.y, 0.0)

    def test__compute_initial_velocity_and_lateral_state(self) -> None:
        """
        This essentially checks that our projection to vehicle/Frenet frame works by reconstructing specified errors.
        """
        current_iteration = SimulationIteration(self.initial_time_point, 0)

        base_initial_state = self.trajectory.get_state_at_time(self.initial_time_point)
        base_pose_rear_axle = base_initial_state.car_footprint.rear_axle

        test_lateral_errors = [-3.0, 3.0]
        test_heading_errors = [-0.1, 0.1]
        test_longitudinal_errors = [-3.0, 3.0]

        error_product = itertools.product(test_lateral_errors, test_heading_errors, test_longitudinal_errors)

        for lateral_error, heading_error, longitudinal_error in error_product:
            theta = base_pose_rear_axle.heading

            delta_x = longitudinal_error * np.cos(theta) - lateral_error * np.sin(theta)
            delta_y = longitudinal_error * np.sin(theta) + lateral_error * np.cos(theta)

            perturbed_pose_rear_axle = StateSE2(
                x=base_pose_rear_axle.x + delta_x,
                y=base_pose_rear_axle.y + delta_y,
                heading=theta + heading_error,
            )

            perturbed_car_footprint = CarFootprint.build_from_rear_axle(
                rear_axle_pose=perturbed_pose_rear_axle,
                vehicle_parameters=base_initial_state.car_footprint.vehicle_parameters,
            )

            perturbed_initial_state = EgoState(
                car_footprint=perturbed_car_footprint,
                dynamic_car_state=base_initial_state.dynamic_car_state,
                tire_steering_angle=base_initial_state.tire_steering_angle,
                is_in_auto_mode=base_initial_state.is_in_auto_mode,
                time_point=base_initial_state.time_point,
            )

            initial_velocity, initial_lateral_state_vector = self.tracker._compute_initial_velocity_and_lateral_state(
                current_iteration=current_iteration,
                initial_state=perturbed_initial_state,
                trajectory=self.trajectory,
            )

            # The velocity should not be impacted by the pose error.
            self.assertEqual(initial_velocity, base_initial_state.dynamic_car_state.rear_axle_velocity_2d.x)

            # We should be able to recover the expected lateral state.  The longitudinal error should have no impact.
            np_test.assert_allclose(
                initial_lateral_state_vector, [lateral_error, heading_error, base_initial_state.tire_steering_angle]
            )

    def test__compute_reference_velocity_and_curvature_profile(self) -> None:
        """
        This test just checks functionality of computing a reference velocity / curvature profile.
        Detailed evaluation of the result is handled in test_tracker_utils and omitted here.
        """
        current_iteration = SimulationIteration(self.initial_time_point, 0)

        reference_velocity, curvature_profile = self.tracker._compute_reference_velocity_and_curvature_profile(
            current_iteration=current_iteration,
            trajectory=self.trajectory,
        )

        tracking_horizon = self.tracker._tracking_horizon
        discretization_time = self.tracker._discretization_time
        lookahead_time_point = TimePoint(
            current_iteration.time_point.time_us + int(1e6 * tracking_horizon * discretization_time)
        )
        expected_lookahead_ego_state = self.trajectory.get_state_at_time(lookahead_time_point)

        # The fitted velocity should match the ego state velocity at least in sign.
        np_test.assert_allclose(
            np.sign(reference_velocity), np.sign(expected_lookahead_ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
        )

        # The curvature profile should have expected shape.
        self.assertEqual(curvature_profile.shape, (tracking_horizon,))

    def test__stopping_controller(self) -> None:
        """Test P controller for when we are coming to a stop."""
        initial_velocity = 5.0  # [m/s]

        # We should apply deceleration if going faster than the reference.
        accel, steering_rate_cmd = self.tracker._stopping_controller(
            initial_velocity=initial_velocity,
            reference_velocity=0.5 * initial_velocity,
        )
        self.assertLess(accel, 0.0)
        self.assertEqual(steering_rate_cmd, 0.0)

        # We should apply positive acceleration if we are moving backwards with a zero reference velocity.
        accel, steering_rate_cmd = self.tracker._stopping_controller(
            initial_velocity=-initial_velocity,
            reference_velocity=0.0,
        )
        self.assertGreater(accel, 0.0)
        self.assertEqual(steering_rate_cmd, 0.0)

    def test__longitudinal_lqr_controller(self) -> None:
        """Test longitudinal control for simple cases of speed above or below the reference velocity."""
        test_initial_velocities = [2.0, 6.0]
        reference_velocity = float(np.mean(test_initial_velocities))

        for initial_velocity in test_initial_velocities:
            # The acceleration command should just act as negative state feedback on velocity error.
            accel_cmd = self.tracker._longitudinal_lqr_controller(
                initial_velocity=initial_velocity, reference_velocity=reference_velocity
            )
            np_test.assert_allclose(np.sign(accel_cmd), -np.sign(initial_velocity - reference_velocity))

    def test__lateral_lqr_controller_straight_road(self) -> None:
        """Test how the controller handles non-zero initial tracking error on a straight road."""
        test_velocity_profile = 5.0 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)  # [m/s]
        test_curvature_profile = 0.0 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)  # [rad/m]

        # Only lateral error.
        test_lateral_errors = [-3.0, 3.0]  # [m]
        for lateral_error in test_lateral_errors:
            initial_lateral_state_vector_lateral_only: npt.NDArray[np.float64] = np.array(
                [lateral_error, 0.0, 0.0], dtype=np.float64
            )
            steering_rate_cmd = self.tracker._lateral_lqr_controller(
                initial_lateral_state_vector=initial_lateral_state_vector_lateral_only,
                velocity_profile=test_velocity_profile,
                curvature_profile=test_curvature_profile,
            )

            # Steering rate should try to cancel lateral error.
            np_test.assert_allclose(
                np.sign(steering_rate_cmd),
                -np.sign(lateral_error),
            )

        # Only heading error.
        test_heading_errors = [-0.1, 0.1]  # [rad]
        for heading_error in test_heading_errors:
            initial_lateral_state_vector_heading_only: npt.NDArray[np.float64] = np.array(
                [0.0, heading_error, 0.0], dtype=np.float64
            )
            steering_rate_cmd = self.tracker._lateral_lqr_controller(
                initial_lateral_state_vector=initial_lateral_state_vector_heading_only,
                velocity_profile=test_velocity_profile,
                curvature_profile=test_curvature_profile,
            )

            # Steering rate should try to cancel heading error.
            np_test.assert_allclose(
                np.sign(steering_rate_cmd),
                -np.sign(heading_error),
            )

    def test__lateral_lqr_controller_curved_road(self) -> None:
        """Test how the controller handles a curved road with zero initial tracking error and zero steering angle."""
        test_velocity_profile = 5.0 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)  # [m/s]
        test_curvature_profile = 0.1 * np.ones(self.tracker._tracking_horizon, dtype=np.float64)  # [rad/m]

        test_initial_lateral_state_vector: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64)

        steering_rate_cmd = self.tracker._lateral_lqr_controller(
            initial_lateral_state_vector=test_initial_lateral_state_vector,
            velocity_profile=test_velocity_profile,
            curvature_profile=test_curvature_profile,
        )

        # Steering rate should try to follow the curvature (same sign).
        np_test.assert_allclose(np.sign(steering_rate_cmd), np.sign(test_curvature_profile[0]))

    def test__solve_one_step_lqr(self) -> None:
        """Test LQR on a simple linear system."""
        # Each state will remain where it is if we apply zero input, so A is the identity.
        # But we have fully controllability with independent control of each state component, reflected in B.
        A: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        B: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        g: npt.NDArray[np.float64] = np.zeros(A.shape[0], dtype=np.float64)  # no affine term in dynamics

        # Cost matrices with equal weight on state tracking and input magnitude.
        Q: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        R: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)

        # We target the origin from a set of non-zero initial states:
        for component_1, component_2 in itertools.product([-5.0, 5.0], [-10.0, 10.0]):
            initial_state: npt.NDArray[np.float64] = np.array([component_1, component_2], dtype=np.float64)

            solution = self.tracker._solve_one_step_lqr(
                initial_state=initial_state,
                reference_state=np.zeros_like(initial_state),
                Q=Q,
                R=R,
                A=A,
                B=B,
                g=g,
                angle_diff_indices=[],
            )

            # The input should be negative state feedback so we cancel out the state component to get close to zero.
            np_test.assert_allclose(np.sign(solution), -np.sign(initial_state))


if __name__ == "__main__":
    unittest.main()
