import unittest

import numpy as np
import numpy.testing as np_test

from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    DoubleMatrix,
    ILQRSolver,
    ILQRSolverParameters,
    ILQRWarmStartParameters,
    _run_lqr_backward_recursion,
)


class TestILQRSolver(unittest.TestCase):
    """
    Tests for ILQRSolver class.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        solver_params = ILQRSolverParameters(
            discretization_time=0.1,
            q_diagonal_entries=[1.0, 1.0, 50.0, 0.0, 0.0],
            r_diagonal_entries=[1e2, 1e4],
            max_ilqr_iterations=100,
            convergence_threshold=0.01,
            alpha_trust_region=0.95,
            min_velocity_linearization=0.01,
            max_acceleration=3.0,
            max_steering_angle_rate=0.5,
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
        self.n_horizon = 20

        self.min_velocity_linearization = self.solver._solver_params.min_velocity_linearization
        self.max_acceleration = self.solver._solver_params.max_acceleration
        self.max_steering_angle_rate = self.solver._solver_params.max_steering_angle_rate

        # Tolerances for np.allclose to pass.
        self.rtol = 1e-8
        self.atol = 1e-10

        # Set up a constant velocity, no turning reference and initial state for testing.
        self.reference_trajectory: DoubleMatrix = np.zeros((self.n_horizon, self.n_states), dtype=np.float64)
        self.reference_trajectory[:, 0] = np.arange(self.n_horizon) * self.discretization_time  # x coordinate
        self.reference_trajectory[:, 3] = 1.0  # constant velocity

    def test__run_lqr_backward_recursion(self) -> None:
        """
        Given dynamics and cost matrices for a controllable but unstable system, ensure that LQR recursion works.
        """
        # Setup dynamics and cost matrices.
        test_a_matrix: DoubleMatrix = (
            self.discretization_time * np.tri(self.n_states, dtype=np.float64).T
        )  # upper triangular
        test_a_matrix[np.diag_indices(self.n_states)] = 1.1

        test_b_matrix: DoubleMatrix = np.zeros((self.n_states, self.n_inputs), dtype=np.float64)
        test_b_matrix[-1, -1] = 1.0

        test_q_matrix: DoubleMatrix = np.eye(self.n_states, dtype=np.float64)
        test_r_matrix: DoubleMatrix = np.eye(self.n_inputs, dtype=np.float64)

        test_a_matrices: DoubleMatrix = np.tile(test_a_matrix, (self.n_horizon, 1, 1))
        test_b_matrices: DoubleMatrix = np.tile(test_b_matrix, (self.n_horizon, 1, 1))
        test_q_matrices: DoubleMatrix = np.tile(test_q_matrix, (self.n_horizon, 1, 1))
        test_r_matrices: DoubleMatrix = np.tile(test_r_matrix, (self.n_horizon, 1, 1))
        test_q_terminal = test_q_matrix

        k_matrices, p_matrices = _run_lqr_backward_recursion(
            test_a_matrices,
            test_b_matrices,
            test_q_matrices,
            test_r_matrices,
            test_q_terminal,
        )

        # Check that feedback gain and value function matrices are the correct shape.
        self.assertEqual(k_matrices.shape, (self.n_horizon, self.n_inputs, self.n_states))
        self.assertEqual(p_matrices.shape, (self.n_horizon, self.n_states, self.n_states))

        # Check that the value function matrix is symmetric and positive semidefinite.
        for p in p_matrices:
            np_test.assert_allclose(p, p.T, atol=self.atol, rtol=self.rtol)
            p_min_eigval = np.amin(np.linalg.eigvals(p))
            self.assertGreaterEqual(p_min_eigval, 0.0)

    def test_solve(self) -> None:
        """
        Test solve calls on a constant velocity reference without turning.
        Note: modification of the reference will invalidate these checks.
        """
        # Case 1: There is no initial tracking error and the reference is constant velocity -> all zero inputs.
        current_state_no_tracking_error: DoubleMatrix = np.copy(self.reference_trajectory[0])
        inputs_no_tracking_error = self.solver.solve(current_state_no_tracking_error, self.reference_trajectory)

        np_test.assert_allclose(inputs_no_tracking_error, 0.0, atol=self.atol, rtol=self.rtol)

        # Case 2: There is some initial tracking error and "" -> not all zero inputs.
        current_state_tracking_error: DoubleMatrix = self.reference_trajectory[0] + [0.1, -0.1, 0.01, 0.5, 0.01]
        inputs_tracking_error = self.solver.solve(current_state_tracking_error, self.reference_trajectory)

        with np_test.assert_raises(AssertionError):
            np_test.assert_allclose(inputs_tracking_error, 0.0, atol=self.atol, rtol=self.rtol)

    def test__clip_inputs(self) -> None:
        """Check that we can clip inputs within constraints."""
        input_sinusoidal_array: DoubleMatrix = np.array(
            [[np.cos(th), np.sin(th)] for th in np.arange(-np.pi, np.pi, 0.1)], dtype=np.float64
        )

        magnitude_within_bounds = [self.max_acceleration, self.max_steering_angle_rate]
        magnitude_outside_bounds = [self.max_acceleration + 1.0, self.max_steering_angle_rate + 0.1]

        for (magnitude, expect_not_modified) in zip([magnitude_within_bounds, magnitude_outside_bounds], [True, False]):
            inputs = magnitude * input_sinusoidal_array
            clipped_inputs = self.solver._clip_inputs(inputs)

            # Confirm input modification matches expectations.
            inputs_are_not_modified = np.allclose(inputs, clipped_inputs, atol=self.atol, rtol=self.rtol)
            self.assertEqual(inputs_are_not_modified, expect_not_modified)

            # Confirm that the clipped inputs lie within bounds.
            clipped_inputs_are_bounded = np.all(np.amax(np.abs(clipped_inputs), axis=0) <= magnitude_within_bounds)
            self.assertTrue(clipped_inputs_are_bounded)

            # Confirm that clipping inputs does not change the sign of inputs.
            np_test.assert_allclose(np.sign(clipped_inputs), np.sign(inputs), atol=self.atol, rtol=self.rtol)

            # Confirm we can apply clipping to an input vector (i.e. single timestep).
            clipped_first_input = self.solver._clip_inputs(inputs[0])
            self.assertTrue(np.all(np.abs(clipped_first_input) <= magnitude_within_bounds))

    def test__input_warm_start(self) -> None:
        """
        Test that we get a reasonable warm start trajectory within bounds when starting on/off the reference trajectory.
        NOTE: This applies for the constant velocity, no turning case - modifications might invalidate these checks.
        """
        # Case 1: The initial state is exactly on the reference, expect all zeros for control input.
        current_state_on_trajectory: DoubleMatrix = np.copy(self.reference_trajectory[0])
        input_warm_start_on_trajectory = self.solver._input_warm_start(
            current_state_on_trajectory, self.reference_trajectory
        )
        np_test.assert_allclose(input_warm_start_on_trajectory, 0.0, atol=self.atol, rtol=self.rtol)

        # Case 2: The initial state is offset from the reference, the control input should not be all zeros.
        current_state_off_trajectory: DoubleMatrix = self.reference_trajectory[0] + [-0.1, 0.1, 0.05, 0.5, 0.1]
        input_warm_start_off_trajectory = self.solver._input_warm_start(
            current_state_off_trajectory, self.reference_trajectory
        )
        with np_test.assert_raises(AssertionError):
            np_test.assert_allclose(input_warm_start_off_trajectory, 0.0, atol=self.atol, rtol=self.rtol)

        # Check conditions that should hold for both cases.
        for input_trajectory in [input_warm_start_on_trajectory, input_warm_start_off_trajectory]:
            # The input warm start trajectory should have expected shape.
            self.assertEqual(input_trajectory.shape, (len(self.reference_trajectory) - 1, self.n_inputs))

            # The input warm start trajectory should be within control input bounds.
            infinity_norm_inputs = np.amax(np.abs(input_trajectory), axis=0)
            warm_start_bounded = np.all(infinity_norm_inputs <= [self.max_acceleration, self.max_steering_angle_rate])
            self.assertTrue(warm_start_bounded)

    def test__update_input_sequence_with_feedback_policy(self) -> None:
        """
        Try to run one iteration of ILQR (so basically just LQR).  If we are above the reference speed,
        the input update should result in higher deceleration.
        """
        # Choose current_state to have some initial tracking error and go faster than the initial reference state.
        current_state: DoubleMatrix = np.copy(self.reference_trajectory[0])
        current_state += [0.1, -0.1, 0.01, 0.5, 0.01]

        input_trajectory = self.solver._input_warm_start(current_state, self.reference_trajectory)
        state_trajectory, state_jacobian_trajectory, input_jacobian_trajectory = self.solver._run_forward_dynamics(
            current_state, input_trajectory
        )

        (
            state_jacobian_matrix_augmented_trajectory,
            input_jacobian_matrix_augmented_trajectory,
        ) = self.solver._linear_augmented_dynamics_matrices(state_jacobian_trajectory, input_jacobian_trajectory)

        (
            state_cost_matrix_augmented_trajectory,
            input_cost_matrix_augmented_trajectory,
            state_cost_matrix_augmented_terminal,
        ) = self.solver._quadratic_augmented_cost_matrices(
            input_trajectory, state_trajectory, self.reference_trajectory
        )

        state_feedback_matrix_augmented_trajectory, _ = _run_lqr_backward_recursion(
            a_matrices=state_jacobian_matrix_augmented_trajectory,
            b_matrices=input_jacobian_matrix_augmented_trajectory,
            q_matrices=state_cost_matrix_augmented_trajectory,
            r_matrices=input_cost_matrix_augmented_trajectory,
            q_terminal=state_cost_matrix_augmented_terminal,
        )

        input_trajectory_next = self.solver._update_input_sequence_with_feedback_policy(
            input_trajectory, state_trajectory, state_feedback_matrix_augmented_trajectory
        )

        self.assertEqual(input_trajectory_next.shape, input_trajectory.shape)

        # Since we are traveling faster than the reference at the initial timestep, we expect acceleration to be lower
        # than the warm start - reflecting the initial speed error.
        self.assertLessEqual(input_trajectory_next[0, 0], input_trajectory[0, 0])

    def test__run_forward_dynamics(self) -> None:
        """Try to run forward dynamics and make sure the results make sense."""
        input_sinusoidal_array: DoubleMatrix = np.array(
            [[np.cos(th), np.sin(th)] for th in np.arange(-np.pi, np.pi, 0.1)], dtype=np.float64
        )
        input_trajectory = [
            self.max_acceleration,
            self.max_steering_angle_rate,
        ] * input_sinusoidal_array

        current_state: DoubleMatrix = np.array([1.0, 5.0, 0.1, 2.0, 0.01], dtype=np.float64)
        state_trajectory, state_jacobian_trajectory, input_jacobian_trajectory = self.solver._run_forward_dynamics(
            current_state, input_trajectory
        )

        # The state trajectory should be 1 longer than the input trajectory and start at z_curr.
        self.assertEqual(len(state_trajectory), 1 + len(input_trajectory))
        np_test.assert_allclose(state_trajectory[0], current_state, atol=self.atol, rtol=self.rtol)

        # The state jacobians should have the same length as the input trajectory and have expected shape.
        self.assertEqual(len(state_jacobian_trajectory), len(input_trajectory))
        self.assertEqual(state_jacobian_trajectory.shape[1:], (self.n_states, self.n_states))

        # The input jacobians should have the same length as the input trajectory and have expected shape.
        self.assertEqual(len(input_jacobian_trajectory), len(input_trajectory))
        self.assertEqual(input_jacobian_trajectory.shape[1:], (self.n_states, self.n_inputs))

    def test__dynamics_and_jacobian(self) -> None:
        """Check that invalid steering angles and near-stopping velocities are handled correctly."""
        current_state_base: DoubleMatrix = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        current_input_base: DoubleMatrix = np.array([1.0, 0.1], dtype=np.float64)

        # Check 1: velocity above threshold should result in unmodified linearization.
        next_state_base, state_jacobian_base, input_jacobian_base = self.solver._dynamics_and_jacobian(
            current_state_base, current_input_base
        )
        velocity_inferred_base = (
            np.hypot(state_jacobian_base[0, 2], state_jacobian_base[1, 2]) / self.discretization_time
        )
        np_test.assert_allclose(velocity_inferred_base, current_state_base[3], atol=self.atol, rtol=self.rtol)

        # We also check the jacobian for this case using finite differencing with an epsilon perturbation.
        epsilon = 1e-6
        state_jacobian_finite_differencing = np.zeros_like(state_jacobian_base)
        for state_idx in range(self.solver._n_states):
            epsilon_array = epsilon * np.array([x == state_idx for x in range(self.solver._n_states)], dtype=np.float64)
            next_state_plus, _, _ = self.solver._dynamics_and_jacobian(
                current_state_base + epsilon_array, current_input_base
            )
            next_state_minus, _, _ = self.solver._dynamics_and_jacobian(
                current_state_base - epsilon_array, current_input_base
            )
            state_jacobian_finite_differencing[:, state_idx] = (next_state_plus - next_state_minus) / (2.0 * epsilon)
        np_test.assert_allclose(state_jacobian_base, state_jacobian_finite_differencing, atol=self.atol, rtol=self.rtol)

        input_jacobian_finite_differencing = np.zeros_like(input_jacobian_base)
        for input_idx in range(self.solver._n_inputs):
            epsilon_array = epsilon * np.array([x == input_idx for x in range(self.solver._n_inputs)], dtype=np.float64)
            next_state_plus, _, _ = self.solver._dynamics_and_jacobian(
                current_state_base, current_input_base + epsilon_array
            )
            next_state_minus, _, _ = self.solver._dynamics_and_jacobian(
                current_state_base, current_input_base - epsilon_array
            )
            input_jacobian_finite_differencing[:, input_idx] = (next_state_plus - next_state_minus) / (2.0 * epsilon)
        np_test.assert_allclose(input_jacobian_base, input_jacobian_finite_differencing, atol=self.atol, rtol=self.rtol)

        # Check 2: velocity below threshold should use a modified linearization.
        current_state_slow: DoubleMatrix = np.copy(current_state_base)
        current_input_slow: DoubleMatrix = np.copy(current_input_base)
        current_state_slow[3] = 0.5 * self.min_velocity_linearization

        _, state_jacobian_slow, _ = self.solver._dynamics_and_jacobian(current_state_slow, current_input_slow)
        velocity_inferred_slow = (
            np.hypot(state_jacobian_slow[0, 2], state_jacobian_slow[1, 2]) / self.discretization_time
        )

        with np_test.assert_raises(AssertionError):
            # The inferred velocity should not be the same as the current state velocity in this case.
            np_test.assert_allclose(velocity_inferred_slow, current_state_slow[3], atol=self.atol, rtol=self.rtol)

        # The inferred velocity should match the min_velocity_linearization in this case.
        np_test.assert_allclose(velocity_inferred_slow, self.min_velocity_linearization, atol=self.atol, rtol=self.rtol)

        # Check 3: steering angle outside bounds should trigger an error.
        current_state_invalid: DoubleMatrix = np.copy(current_state_base)
        current_input_invalid: DoubleMatrix = np.copy(current_input_base)
        current_state_invalid[4] = np.pi / 2.0
        with self.assertRaises(AssertionError):
            self.solver._dynamics_and_jacobian(current_state_invalid, current_input_invalid)

    def test__linear_augmented_dynamics_matrices(self) -> None:
        """Test that lifting the LTV system matrices is handled correctly."""
        test_state_jacobian: DoubleMatrix = np.arange(self.n_states**2, dtype=np.float64).reshape(
            self.n_states, self.n_states
        )
        test_input_jacobian: DoubleMatrix = np.arange(self.n_states * self.n_inputs, dtype=np.float64).reshape(
            self.n_states, self.n_inputs
        )

        test_state_jacobians: DoubleMatrix = np.tile(test_state_jacobian, (self.n_horizon, 1, 1))
        test_input_jacobians: DoubleMatrix = np.tile(test_input_jacobian, (self.n_horizon, 1, 1))

        state_jacobians_augmented, input_jacobians_augmented = self.solver._linear_augmented_dynamics_matrices(
            test_state_jacobians, test_input_jacobians
        )

        # The number of matrices should not have changed.
        self.assertEqual(len(state_jacobians_augmented), self.n_horizon)
        self.assertEqual(len(input_jacobians_augmented), self.n_horizon)

        for matrix_augmented, matrix_original in zip(state_jacobians_augmented, test_state_jacobians):
            # The first block should match the original matrix.
            np_test.assert_allclose(
                matrix_augmented[: self.n_states, : self.n_states], matrix_original, atol=self.atol, rtol=self.rtol
            )

            # Last row/column should be all zeros aside for the very last entry.
            np_test.assert_allclose(matrix_augmented[-1, :-1], 0.0, atol=self.atol, rtol=self.rtol)
            np_test.assert_allclose(matrix_augmented[:-1, -1], 0.0, atol=self.atol, rtol=self.rtol)
            np_test.assert_allclose(matrix_augmented[-1, -1], 1.0, atol=self.atol, rtol=self.rtol)

        for matrix_augmented, matrix_original in zip(input_jacobians_augmented, test_input_jacobians):
            # The first block should match the original matrix.
            np_test.assert_allclose(
                matrix_augmented[: self.n_states, : self.n_inputs], matrix_original, atol=self.atol, rtol=self.rtol
            )

            # Last row/column should be all zeros.
            np_test.assert_allclose(matrix_augmented[-1, :], 0.0, atol=self.atol, rtol=self.rtol)
            np_test.assert_allclose(matrix_augmented[:, -1], 0.0, atol=self.atol, rtol=self.rtol)

    def test__quadratic_augmented_cost_matrices(self) -> None:
        """
        Test that the LTV system quadratic cost matrices are consistent with expectations.
        """
        current_state: DoubleMatrix = np.copy(self.reference_trajectory[0])
        current_state += [0.1, -0.1, 0.01, 0.5, 0.01]

        input_warm_start = self.solver._input_warm_start(current_state, self.reference_trajectory)
        state_warm_start, _, _ = self.solver._run_forward_dynamics(current_state, input_warm_start)

        (
            state_cost_matrix_augmented_trajectory,
            input_cost_matrix_augmented_trajectory,
            state_cost_matrix_augmented_terminal,
        ) = self.solver._quadratic_augmented_cost_matrices(
            input_warm_start, state_warm_start, self.reference_trajectory
        )

        self.assertEqual(len(state_cost_matrix_augmented_trajectory), len(input_warm_start))
        self.assertEqual(len(input_cost_matrix_augmented_trajectory), len(input_warm_start))

        # Check positive semidefiniteness of cost matrices.  The input cost may not be positive semidefinite due to
        # lifting (i.e. we're using an augmented state).
        for (q, r) in zip(state_cost_matrix_augmented_trajectory, input_cost_matrix_augmented_trajectory):
            min_eigval_q = np.amin(np.linalg.eigvals(q))
            self.assertGreaterEqual(min_eigval_q, 0.0)
            np_test.assert_allclose(q, q.T, atol=self.atol, rtol=self.rtol)

            min_eigval_r = np.amin(np.linalg.eigvals(r))
            self.assertGreaterEqual(min_eigval_r, 0.0)
            np_test.assert_allclose(r, r.T, atol=self.atol, rtol=self.rtol)

        q_terminal = state_cost_matrix_augmented_terminal
        min_eigval_q_terminal = np.amin(np.linalg.eigvals(q_terminal))
        self.assertGreaterEqual(min_eigval_q_terminal, 0.0)
        np_test.assert_allclose(q_terminal, q_terminal.T, atol=self.atol, rtol=self.rtol)

        # The initial state cost matrix is a special case where we should only be applying the trust region penalty
        # and no tracking error cost.
        q_initial = state_cost_matrix_augmented_trajectory[0]
        alpha_trust_region = self.solver._solver_params.alpha_trust_region
        np_test.assert_allclose(
            q_initial[:-1, :-1], alpha_trust_region * np.eye(self.n_states), atol=self.atol, rtol=self.rtol
        )
        np_test.assert_allclose(q_initial[-1, :], 0.0, atol=self.atol, rtol=self.rtol)
        np_test.assert_allclose(q_initial[:, -1], 0.0, atol=self.atol, rtol=self.rtol)


if __name__ == "__main__":
    unittest.main()
