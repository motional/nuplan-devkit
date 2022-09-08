import time
import unittest
from functools import partial

import numpy as np
import numpy.testing as np_test

from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    DoubleMatrix,
    ILQRInputPolicy,
    ILQRIterate,
    ILQRSolver,
    ILQRSolverParameters,
    ILQRWarmStartParameters,
)


class TestILQRSolver(unittest.TestCase):
    """
    Tests for ILQRSolver class.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        solver_params = ILQRSolverParameters(
            discretization_time=0.2,
            state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],
            input_cost_diagonal_entries=[1.0, 10.0],
            state_trust_region_entries=[1.0] * 5,
            input_trust_region_entries=[1.0] * 2,
            max_ilqr_iterations=100,
            convergence_threshold=1e-6,
            max_solve_time=0.05,
            max_acceleration=3.0,
            max_steering_angle=np.pi / 3.0,
            max_steering_angle_rate=0.5,
            min_velocity_linearization=0.01,
        )

        warm_start_params = ILQRWarmStartParameters(
            k_velocity_error_feedback=0.5,
            k_steering_angle_error_feedback=0.05,
            lookahead_distance_lateral_error=15.0,
            k_lateral_error=0.1,
            jerk_penalty_warm_start_fit=1e-4,
            curvature_rate_penalty_warm_start_fit=1e-2,
        )

        self.solver = ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params)

        self.discretization_time = self.solver._solver_params.discretization_time
        self.n_states = self.solver._n_states
        self.n_inputs = self.solver._n_inputs
        self.n_horizon = 40

        self.min_velocity_linearization = self.solver._solver_params.min_velocity_linearization
        self.max_acceleration = self.solver._solver_params.max_acceleration
        self.max_steering_angle = self.solver._solver_params.max_steering_angle
        self.max_steering_angle_rate = self.solver._solver_params.max_steering_angle_rate

        # Tolerances for np.allclose to pass.
        self.rtol = 1e-8
        self.atol = 1e-10

        # Wrappers for allclose/assert_allclose that use specified tolerances.
        self.check_if_allclose = partial(np.allclose, rtol=self.rtol, atol=self.atol)
        self.assert_allclose = partial(np_test.assert_allclose, rtol=self.rtol, atol=self.atol)

        # Set up a constant velocity, no turning reference and initial state for testing.
        constant_speed = 1.0
        self.reference_trajectory: DoubleMatrix = np.zeros((self.n_horizon + 1, self.n_states), dtype=np.float64)
        self.reference_trajectory[:, 0] = constant_speed * np.arange(self.n_horizon + 1) * self.discretization_time
        self.reference_trajectory[:, 3] = constant_speed

        reference_acceleration = np.diff(self.reference_trajectory[:, 3]) / self.discretization_time
        reference_steering_rate = np.diff(self.reference_trajectory[:, 4]) / self.discretization_time
        self.reference_inputs: DoubleMatrix = np.column_stack((reference_acceleration, reference_steering_rate))

    def test_solve(self) -> None:
        """
        Check that, for reasonable tuning parameters and small initial error, the tracking cost is non-increasing.
        """
        perturbed_current_state: DoubleMatrix = np.array([1.0, -1.0, 0.01, 1.0, -0.1], dtype=np.float64)
        current_state = self.reference_trajectory[0, :] + perturbed_current_state

        start_time = time.perf_counter()
        ilqr_solutions = self.solver.solve(current_state, self.reference_trajectory)
        end_time = time.perf_counter()

        print(f"Solve took {end_time - start_time} seconds for {len(ilqr_solutions)} iterations.")

        tracking_cost_history = [isol.tracking_cost for isol in ilqr_solutions]

        self.assertTrue(np.all(np.diff(tracking_cost_history) <= 0.0))

    def test__compute_tracking_cost(self) -> None:
        """Check tracking cost computation."""
        zero_input_trajectory: DoubleMatrix = np.zeros((self.n_horizon, self.n_inputs), dtype=np.float64)
        zero_state_jacobian_trajectory: DoubleMatrix = np.zeros(
            (self.n_horizon, self.n_states, self.n_states), dtype=np.float64
        )
        zero_input_jacobian_trajectory: DoubleMatrix = np.zeros(
            (self.n_horizon, self.n_states, self.n_inputs), dtype=np.float64
        )

        test_iterate = ILQRIterate(
            input_trajectory=np.zeros((self.n_horizon, self.n_inputs), dtype=np.float64),
            state_trajectory=self.reference_trajectory,
            state_jacobian_trajectory=zero_state_jacobian_trajectory,
            input_jacobian_trajectory=zero_input_jacobian_trajectory,
        )

        cost_zero_error_zero_inputs = self.solver._compute_tracking_cost(
            iterate=test_iterate,
            reference_trajectory=self.reference_trajectory,
        )

        self.assert_allclose(0.0, cost_zero_error_zero_inputs)

        # Larger norm values of inputs should increase cost, regardless of the sign.
        for input_sign in [-1.0, 1.0]:
            costs = []
            for input_magnitude in [1.0, 5.0, 10.0, 100.0]:
                test_input_trajectory = (
                    input_sign * input_magnitude * np.ones((self.n_horizon, self.n_inputs), dtype=np.float64)
                )

                test_iterate = ILQRIterate(
                    input_trajectory=test_input_trajectory,
                    state_trajectory=self.reference_trajectory,
                    state_jacobian_trajectory=zero_state_jacobian_trajectory,
                    input_jacobian_trajectory=zero_input_jacobian_trajectory,
                )

                cost = self.solver._compute_tracking_cost(
                    iterate=test_iterate,
                    reference_trajectory=self.reference_trajectory,
                )

                costs.append(cost)

            self.assertTrue(np.all(np.diff(costs) > 0.0))

        # Larger norm deviations in state trajectory should increase cost, regardless of the sign.
        for error_sign in [-1.0, 1.0]:
            costs = []
            for error_magnitude in [1.0, 5.0, 10.0, 100.0]:
                test_state_trajectory = self.reference_trajectory + error_sign * error_magnitude * np.ones_like(
                    self.reference_trajectory
                )

                test_iterate = ILQRIterate(
                    input_trajectory=zero_input_trajectory,
                    state_trajectory=test_state_trajectory,
                    state_jacobian_trajectory=zero_state_jacobian_trajectory,
                    input_jacobian_trajectory=zero_input_jacobian_trajectory,
                )

                cost = self.solver._compute_tracking_cost(
                    iterate=test_iterate,
                    reference_trajectory=self.reference_trajectory,
                )

                costs.append(cost)

            self.assertTrue(np.all(np.diff(costs) > 0.0))

    def test__clip_inputs(self) -> None:
        """Check that input clipping works."""
        zero_inputs: DoubleMatrix = np.zeros(self.n_inputs, dtype=np.float64)
        inputs_at_bounds: DoubleMatrix = np.array(
            [self.max_acceleration, self.max_steering_angle_rate], dtype=np.float64
        )
        inputs_within_bounds = 0.5 * inputs_at_bounds
        inputs_outside_bounds = 2.0 * inputs_at_bounds

        for test_inputs, expect_same in zip(
            [zero_inputs, inputs_within_bounds, inputs_at_bounds, inputs_outside_bounds], [True, True, True, False]
        ):
            for input_sign in [-1.0, 1.0]:
                signed_test_inputs = input_sign * test_inputs
                clipped_inputs = self.solver._clip_inputs(signed_test_inputs)
                are_same = self.check_if_allclose(signed_test_inputs, clipped_inputs)

                self.assertEqual(are_same, expect_same)

                self.assertTrue(np.all(np.sign(signed_test_inputs) == np.sign(clipped_inputs)))

    def test__clip_steering_angle(self) -> None:
        """Check that steering angle clipping works."""
        zero_steering_angle = 0.0
        steering_angle_at_bounds = self.max_steering_angle
        steering_angle_within_bounds = 0.5 * steering_angle_at_bounds
        steering_angle_outside_bounds = 2.0 * steering_angle_at_bounds

        for test_steering_angle, expect_same in zip(
            [
                zero_steering_angle,
                steering_angle_within_bounds,
                steering_angle_at_bounds,
                steering_angle_outside_bounds,
            ],
            [True, True, True, False],
        ):
            for sign in [-1.0, 1.0]:
                signed_steering_angle = sign * test_steering_angle
                clipped_steering_angle = self.solver._clip_steering_angle(signed_steering_angle)
                are_same = self.check_if_allclose(signed_steering_angle, clipped_steering_angle)

                self.assertEqual(are_same, expect_same)

                self.assertTrue(np.all(np.sign(signed_steering_angle) == np.sign(clipped_steering_angle)))

    def test__input_warm_start(self) -> None:
        """Check first warm start generation under zero and nonzero initial tracking error."""
        test_current_state = self.reference_trajectory[0, :]
        warm_start_iterate = self.solver._input_warm_start(test_current_state, self.reference_trajectory)
        self.assert_allclose(warm_start_iterate.input_trajectory, self.reference_inputs)

        # Check that we apply feedback with just the first input in the warm start.
        perturbed_current_state = self.reference_trajectory[0, :] + np.array(
            [1.0, -1.0, 0.01, 1.0, -0.1], dtype=np.float64
        )
        perturbed_warm_start_iterate = self.solver._input_warm_start(perturbed_current_state, self.reference_trajectory)
        first_input_close = self.check_if_allclose(
            perturbed_warm_start_iterate.input_trajectory[0], self.reference_inputs[0]
        )
        self.assertFalse(first_input_close)
        self.assert_allclose(perturbed_warm_start_iterate.input_trajectory[1, :], self.reference_inputs[1, :])

    def test__run_forward_dynamics_no_saturation(self) -> None:
        """Check generation of a state trajectory from current state and inputs without steering angle saturation."""
        test_current_state = self.reference_trajectory[0, :]
        current_steering_angle = test_current_state[4]

        # We don't want to saturate the steering angle and thus need to know how long this will take.
        time_to_saturation_s = (self.max_steering_angle - abs(current_steering_angle)) / self.max_steering_angle_rate
        timesteps_to_saturation = np.ceil(time_to_saturation_s / self.discretization_time).astype(int)
        assert timesteps_to_saturation >= 2, "We'd like at least two timesteps_to_saturation for the subsequent test."

        steering_rate_input = self.max_steering_angle_rate
        acceleration_input = self.max_acceleration

        # We pick a trajectory length short enough to avoid steering angle saturation.
        test_input_trajectory: DoubleMatrix = np.ones((timesteps_to_saturation - 1, 2), dtype=np.float64)
        test_input_trajectory[:, 0] = acceleration_input
        test_input_trajectory[:, 1] = steering_rate_input

        ilqr_iterate = self.solver._run_forward_dynamics(test_current_state, test_input_trajectory)

        # Check 1: We did not saturate the steering angle - i.e. we never hit any of the limits.
        self.assertLess(np.amax(np.abs(ilqr_iterate.state_trajectory[:, 4])), self.max_steering_angle)

        # Check 2: We should not see any modification of steering rate.
        steering_rate_unmodified = self.check_if_allclose(
            ilqr_iterate.input_trajectory[:, 1], test_input_trajectory[:, 1]
        )
        self.assertTrue(steering_rate_unmodified)

        # Check 3: Trivial check that final inputs are consistent with velocity and steering angle states.
        acceleration_finite_differences = np.diff(ilqr_iterate.state_trajectory[:, 3]) / self.discretization_time
        self.assert_allclose(acceleration_finite_differences, ilqr_iterate.input_trajectory[:, 0])

        steering_rate_finite_differences = np.diff(ilqr_iterate.state_trajectory[:, 4]) / self.discretization_time
        self.assert_allclose(steering_rate_finite_differences, ilqr_iterate.input_trajectory[:, 1])

    def test__run_forward_dynamics_saturation(self) -> None:
        """Check generation of a state trajectory from current state and inputs with steering angle saturation."""
        test_current_state = self.reference_trajectory[0, :]
        current_steering_angle = test_current_state[4]

        # We try to saturate the steering angle and thus need to know how long this will take.
        time_to_saturation_s = (self.max_steering_angle - abs(current_steering_angle)) / self.max_steering_angle_rate
        timesteps_to_saturation = np.ceil(time_to_saturation_s / self.discretization_time).astype(int)
        assert timesteps_to_saturation >= 2, "We'd like at least two timesteps_to_saturation for the subsequent test."

        if np.abs(current_steering_angle) > 0.0:
            # Keep the steering rate sign consistent with the current steering angle.
            steering_rate_input = self.max_steering_angle_rate * np.sign(current_steering_angle)
        else:
            # We have zero current steering angle so just opt to reach +self.max_steering_angle.
            steering_rate_input = self.max_steering_angle_rate

        acceleration_input = self.max_acceleration

        # Make an input trajectory with maxed inputs that is long enough to achieve saturation.
        test_input_trajectory: DoubleMatrix = np.ones((timesteps_to_saturation + 1, 2), dtype=np.float64)
        test_input_trajectory[:, 0] = acceleration_input
        test_input_trajectory[:, 1] = steering_rate_input

        ilqr_iterate = self.solver._run_forward_dynamics(test_current_state, test_input_trajectory)

        # Check 1: We did actually saturate the steering angle - i.e. we hit the max by the penultimate timestep
        #          and never went out of the limits.
        self.assertEqual(np.abs(ilqr_iterate.state_trajectory[-2, 4]), self.max_steering_angle)
        self.assertEqual(np.amax(np.abs(ilqr_iterate.state_trajectory[:, 4])), self.max_steering_angle)

        # Check 2: We see a resulting impact on the steering rate input - the applied doesn't match the test values.
        steering_rate_unmodified = self.check_if_allclose(
            ilqr_iterate.input_trajectory[:, 1], test_input_trajectory[:, 1]
        )
        self.assertFalse(steering_rate_unmodified)

        # Check 3: Trivial check that final inputs are consistent with velocity and steering angle states.
        acceleration_finite_differences = np.diff(ilqr_iterate.state_trajectory[:, 3]) / self.discretization_time
        steering_rate_finite_differences = np.diff(ilqr_iterate.state_trajectory[:, 4]) / self.discretization_time

        self.assert_allclose(acceleration_finite_differences, ilqr_iterate.input_trajectory[:, 0])
        self.assert_allclose(steering_rate_finite_differences, ilqr_iterate.input_trajectory[:, 1])

    def test_dynamics_and_jacobian_constraints(self) -> None:
        """Check application of constraints in dynamics."""
        test_state_in_bounds: DoubleMatrix = np.array([0.0, 0.0, 0.1, 1.0, -0.01], dtype=np.float64)

        # Test state where the steering angle is at its maximum value.
        test_state_at_bounds: DoubleMatrix = np.copy(test_state_in_bounds)
        test_state_at_bounds[4] = self.max_steering_angle

        input_at_bounds: DoubleMatrix = np.array(
            [self.max_acceleration, self.max_steering_angle_rate], dtype=np.float64
        )

        # Valid and invalid test inputs relative to input constraints.
        test_input_in_bounds = 0.5 * input_at_bounds
        test_input_outside_bounds = 2.0 * input_at_bounds

        test_cases_dict = {}
        test_cases_dict["state_in_bounds_input_in_bounds"] = {
            "state": test_state_in_bounds,
            "input": test_input_in_bounds,
            "expect_acceleration_modified": False,
            "expect_steering_rate_modified": False,
        }
        test_cases_dict["state_at_bounds_input_in_bounds"] = {
            "state": test_state_at_bounds,
            "input": test_input_in_bounds,
            "expect_acceleration_modified": False,
            "expect_steering_rate_modified": True,
        }
        test_cases_dict["state_in_bounds_input_outside_bounds"] = {
            "state": test_state_in_bounds,
            "input": test_input_outside_bounds,
            "expect_acceleration_modified": True,
            "expect_steering_rate_modified": True,
        }
        test_cases_dict["state_at_bounds_input_outside_bounds"] = {
            "state": test_state_at_bounds,
            "input": test_input_outside_bounds,
            "expect_acceleration_modified": True,
            "expect_steering_rate_modified": True,
        }

        for test_name, test_config in test_cases_dict.items():
            next_state, applied_input, _, _ = self.solver._dynamics_and_jacobian(
                test_config["state"], test_config["input"]
            )

            # There can be variation between applied_input and the function argument input due to constraints.
            # We check that constraints are correctly imposed by seeing if the input was modified and if it's expected.
            self.assertEqual(
                test_config["expect_acceleration_modified"],
                not self.check_if_allclose(applied_input[0], test_config["input"][0]),
            )
            self.assertEqual(
                test_config["expect_steering_rate_modified"],
                not self.check_if_allclose(applied_input[1], test_config["input"][1]),
            )

            # The steering angle of the next state must always be feasible.
            self.assertLessEqual(np.abs(next_state[4]), self.max_steering_angle)

    def test_dynamics_and_jacobian_linearization(self) -> None:
        """
        Check that Jacobian computation makes sense by comparison to finite difference estimate.
        Also check that the minimum velocity linearization is triggered for the Jacobian computation.
        """
        test_state: DoubleMatrix = np.array([0.0, 0.0, 0.1, 1.0, -0.01], dtype=np.float64)
        test_input: DoubleMatrix = np.array([1.0, 0.01], dtype=np.float64)
        epsilon = 1e-6  # A small positive number used for finite difference computation.

        _, applied_input, state_jacobian, input_jacobian = self.solver._dynamics_and_jacobian(test_state, test_input)

        # We expect that the applied_input should be the same as the test_input, else constraints were applied.
        # In this case, finite differencing to estimate the Jacobian won't be accurate.
        self.assert_allclose(test_input, applied_input)

        # State Jacobian check.
        state_jacobian_finite_differencing = np.zeros_like(state_jacobian)
        for state_idx in range(self.n_states):
            epsilon_array = epsilon * np.array([x == state_idx for x in range(self.n_states)], dtype=np.float64)
            next_state_plus, _, _, _ = self.solver._dynamics_and_jacobian(test_state + epsilon_array, test_input)
            next_state_minus, _, _, _ = self.solver._dynamics_and_jacobian(test_state - epsilon_array, test_input)
            state_jacobian_finite_differencing[:, state_idx] = (next_state_plus - next_state_minus) / (2.0 * epsilon)
        self.assert_allclose(state_jacobian, state_jacobian_finite_differencing)

        # Input Jacobian check.
        input_jacobian_finite_differencing = np.zeros_like(input_jacobian)
        for input_idx in range(self.n_inputs):
            epsilon_array = epsilon * np.array([x == input_idx for x in range(self.n_inputs)], dtype=np.float64)
            next_state_plus, _, _, _ = self.solver._dynamics_and_jacobian(test_state, test_input + epsilon_array)
            next_state_minus, _, _, _ = self.solver._dynamics_and_jacobian(test_state, test_input - epsilon_array)
            input_jacobian_finite_differencing[:, input_idx] = (next_state_plus - next_state_minus) / (2.0 * epsilon)
        self.assert_allclose(input_jacobian, input_jacobian_finite_differencing)

        # Check that we apply the minimum velocity linearization threshold - i.e. we don't use 0 for the state Jacobian.
        test_state_stopped: DoubleMatrix = np.copy(test_state)
        test_state_stopped[3] = 0.0

        _, _, state_jacobian_stopped, _ = self.solver._dynamics_and_jacobian(test_state_stopped, test_input)
        velocity_inferred_stopped = (
            np.hypot(state_jacobian_stopped[0, 2], state_jacobian_stopped[1, 2]) / self.discretization_time
        )
        with np_test.assert_raises(AssertionError):
            self.assert_allclose(velocity_inferred_stopped, test_state_stopped[3])
        self.assert_allclose(velocity_inferred_stopped, self.min_velocity_linearization)

    def test__run_lqr_backward_recursion(self) -> None:
        """Check some properties of the LQR input policy."""
        test_current_state = self.reference_trajectory[0]
        test_input_trajectory = self.reference_inputs

        # Add an initial input perturbation to the reference inputs to introduce tracking error.
        input_perturbation: DoubleMatrix = np.array(
            [-0.1 * self.max_acceleration, 0.1 * self.max_steering_angle], dtype=np.float64
        )
        test_input_trajectory[0] += input_perturbation

        ilqr_iterate = self.solver._run_forward_dynamics(test_current_state, test_input_trajectory)

        ilqr_input_policy = self.solver._run_lqr_backward_recursion(
            current_iterate=ilqr_iterate,
            reference_trajectory=self.reference_trajectory,
        )

        state_feedback_matrices = ilqr_input_policy.state_feedback_matrices
        feedforward_inputs = ilqr_input_policy.feedforward_inputs

        # Check 1: We expect the feedforward input to somewhat compensate for the initial input perturbation.
        # It basically should act like negative feedback so we expect the sign to oppose the perturbation.
        self.assertTrue(np.all(np.sign(feedforward_inputs[0]) == -np.sign(input_perturbation)))

        # Check 2: We just simply check the shape of the state feedback matrix - numerical checks are somewhat flaky.
        self.assertEqual(state_feedback_matrices.shape, (self.n_horizon, self.n_inputs, self.n_states))

    def test__update_inputs_with_policy(self) -> None:
        """Check how application of a specified input policy affects the next input trajectory."""
        ilqr_iterate = self.solver._run_forward_dynamics(self.reference_trajectory[0], self.reference_inputs)
        input_trajectory = ilqr_iterate.input_trajectory
        state_trajectory = ilqr_iterate.state_trajectory

        # Check 1: Test feedforward inputs alone, applying them without saturation should be reflected in next inputs.
        angle_distance_to_saturation = self.max_steering_angle - np.amax(np.abs(state_trajectory[:, 4]))
        test_feedforward_steering_rate = min(
            0.5 * angle_distance_to_saturation / (self.n_horizon * self.discretization_time),
            self.max_steering_angle_rate,
        )

        feedforward_inputs = np.ones_like(input_trajectory)
        feedforward_inputs[:, 0] = self.max_acceleration
        feedforward_inputs[:, 1] = test_feedforward_steering_rate
        feedforward_inputs[1::2] *= -1.0  # make this oscillatory so integral is 0.

        state_feedback_matrices = np.zeros((len(feedforward_inputs), self.n_inputs, self.n_states))

        lqr_input_policy = ILQRInputPolicy(
            state_feedback_matrices=state_feedback_matrices, feedforward_inputs=feedforward_inputs
        )

        input_next_trajectory = self.solver._update_inputs_with_policy(
            current_iterate=ilqr_iterate,
            lqr_input_policy=lqr_input_policy,
        )

        self.assert_allclose(feedforward_inputs, input_next_trajectory)

        # Check 2: Suppose we apply an initial non-zero feedforward input perturbation with negative state feedback
        #          matrices.  The second input should try to counteract the perturbation applied in the first.
        test_input_perturbation = [self.max_acceleration, test_feedforward_steering_rate]
        feedforward_inputs = np.zeros_like(input_trajectory)
        feedforward_inputs[0, :] = test_input_perturbation

        state_feedback_matrices = np.zeros((len(feedforward_inputs), self.n_inputs, self.n_states))
        state_feedback_matrices[:, 0, 3] = -1.0  # i.e. acceleration is negative feedback on velocity.
        state_feedback_matrices[:, 1, 4] = -1.0  # i.e. steering rate is negative feedback on steering angle.

        lqr_input_policy = ILQRInputPolicy(
            state_feedback_matrices=state_feedback_matrices, feedforward_inputs=feedforward_inputs
        )

        input_next_trajectory = self.solver._update_inputs_with_policy(
            current_iterate=ilqr_iterate,
            lqr_input_policy=lqr_input_policy,
        )

        first_delta_input = input_next_trajectory[0, :] - input_trajectory[0, :]
        second_delta_input = input_next_trajectory[1, :] - input_trajectory[1, :]

        self.assertTrue(np.all(np.sign(first_delta_input) == -np.sign(second_delta_input)))


if __name__ == "__main__":
    unittest.main()
