"""
This provides an implementation of the iterative linear quadratic regulator (iLQR) algorithm for trajectory tracking.
It is specialized to the case where we operate in discrete time with a kinematic bicycle model
and a quadratic trajectory tracking cost.

Original (Nonlinear) Discrete Time System:
    z_k = [x_k, y_k, theta_k, v_k, delta_k]
    u_k = [a_k, phi_k]

    x_{k+1}     = x_k     + v_k * cos(theta_k) * dt
    y_{k+1}     = y_k     + v_k * sin(theta_k) * dt
    theta_{k+1} = theta_k + v_k * tan(phi_k) / L * dt
    v_{k+1}     = v_k     + a_k * dt
    delta_{k+1} = delta_k + phi_k * dt

    where (x_k, y_k, theta_k) is the pose at timestep k with time discretization dt,
    v_k and a_k are velocity and acceleration,
    delta_k and phi_k are steering angle and steering angle rate,
    and L is the vehicle wheelbase.

Quadratic Tracking Cost:
    J = sum_{k=0}^{N-1} || u_k ||_2^R +
        sum_{k=1}^N || z_k - z_{ref,k} ||_2^Q

Reference Used: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec5-LQR.pdf
Implementation/Derivation Page: https://confluence.ci.motional.com/confluence/x/VaOdCg
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.compute import principal_value
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    complete_kinematic_state_and_inputs_from_poses,
    compute_steering_angle_feedback,
)

DoubleMatrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ILQRSolverParameters:
    """Parameters related to the solver implementation of iLQR."""

    discretization_time: float  # [s] Time discretization used for integration.
    q_diagonal_entries: List[float]  # Cost weights for state variables [x, y, heading, velocity, steering angle]
    r_diagonal_entries: List[float]  # Cost weights for input variables [acceleration, steering rate]
    max_ilqr_iterations: int  # Maximum number of iterations to run iLQR before timeout.
    convergence_threshold: float  # Threshold for delta inputs below which we can terminate iLQR early.
    alpha_trust_region: float  # Trust region parameter in (0.0, 1.0).  Used to keep linearization error bounded.
    min_velocity_linearization: float  # [m/s] Absolute value threshold below which linearization velocity is modified.
    max_acceleration: float  # [m/s^2] Absolute value threshold on acceleration input.
    max_steering_angle_rate: float  # [rad/s] Absolute value threshold on steering rate input.
    wheelbase: float = get_pacifica_parameters().wheel_base  # [m] Wheelbase length parameter for the vehicle.

    def __post_init__(self) -> None:
        """Ensure entries lie in expected bounds and initialize wheelbase."""
        for entry in [
            "discretization_time",
            "max_ilqr_iterations",
            "convergence_threshold",
            "min_velocity_linearization",
            "max_acceleration",
            "max_steering_angle_rate",
            "wheelbase",
        ]:
            assert getattr(self, entry) > 0.0, f"Field {entry} should be positive."

        assert self.alpha_trust_region > 0.0, "Trust region alpha should be positive."
        assert self.alpha_trust_region < 1.0, "Trust region alpha should be in (0.0, 1.0)."

        assert np.all([x >= 0 for x in self.q_diagonal_entries]), "Q matrix must be positive semidefinite."
        assert np.all([x > 0 for x in self.r_diagonal_entries]), "R matrix must be positive definite."


@dataclass(frozen=True)
class ILQRWarmStartParameters:
    """Parameters related to generating a warm start trajectory for iLQR."""

    k_velocity_error_feedback: float  # Gain for initial velocity error for warm start acceleration.
    k_steering_angle_error_feedback: float  # Gain for initial steering angle error for warm start steering rate.
    lookahead_distance_lateral_error: float  # [m] Distance ahead for which we estimate lateral error.
    k_lateral_error: float  # Gain for lateral error to compute steering angle feedback.
    jerk_penalty_warm_start_fit: float  # Penalty for jerk in velocity profile estimation.
    curvature_rate_penalty_warm_start_fit: float  # Penalty for curvature rate in curvature profile estimation.

    def __post_init__(self) -> None:
        """Ensure entries lie in expected bounds."""
        for entry in [
            "k_velocity_error_feedback",
            "k_steering_angle_error_feedback",
            "lookahead_distance_lateral_error",
            "k_lateral_error",
            "jerk_penalty_warm_start_fit",
            "curvature_rate_penalty_warm_start_fit",
        ]:
            assert getattr(self, entry) > 0.0, f"Field {entry} should be positive."


def _run_lqr_backward_recursion(
    a_matrices: DoubleMatrix,
    b_matrices: DoubleMatrix,
    q_matrices: DoubleMatrix,
    r_matrices: DoubleMatrix,
    q_terminal: DoubleMatrix,
) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Given a linear time varying (LTV) system with dynamics matrices {A_k}, {B_k} and cost matrices {Q_k}, {R_k},
    compute the optimal LQR state feedback using dynamic programming / backward recursion.
    We opt to use generic math notation to emphasize this applies generally and not just for the bicycle model.
    The state index k runs from 0 to N-1.  The terminal state occurs at state index N.
    :param a_matrices: A N-collection of n_states x n_states matrices for the LTV system.
    :param b matrices: A N-collection of n_states x n_inputs matrices for the LTV system.
    :param q_matrices: A N-collection of n_states x n_states matrices representing state stage cost.
    :param r_matrices: A N-collection of n_inputs x n_inputs matrices representing input stage cost.
    :param q_terminal: A terminal state cost matrix (n_states x n_states) from which we start the recursion.
    :return: The optimal LQR state feedback matrices {K_k} and corresponding value function matrix {P_k}, k=0^{N-1}.
    """
    # Matrix Length Check.
    assert len(a_matrices) == len(b_matrices), "Dynamics matrices should have the same length."
    assert len(q_matrices) == len(r_matrices), "Cost matrices should have the same length."
    assert len(a_matrices) == len(q_matrices), "Dynamics and cost matrices should have the same length."

    # Matrix Size Check.
    state_dimension = a_matrices[0].shape[0]  # inferred state dimension
    input_dimension = b_matrices[0].shape[1]  # inferred input dimension

    for (a, b, q, r) in zip(a_matrices, b_matrices, q_matrices, r_matrices):
        assert a.shape == (state_dimension, state_dimension), "Invalid state matrix shape."
        assert b.shape == (state_dimension, input_dimension), "Invalid input matrix shape."
        assert q.shape == (state_dimension, state_dimension), "Invalid state cost matrix shape."
        assert r.shape == (input_dimension, input_dimension), "Invalid input cost matrix shape."

    assert q_terminal.shape == (state_dimension, state_dimension), "Invalid terminal state cost matrix."

    p_current = q_terminal  # Initiate the recursion with P_N = Q_N, terminal value function matrix.
    horizon_length = len(a_matrices)  # Horizon length N.

    k_matrices: List[DoubleMatrix] = []  # Optimal state feedback gain matrices found using LQR.
    p_matrices: List[DoubleMatrix] = []  # Optimal value function matrices found using LQR.

    for idx in reversed(range(horizon_length)):
        a = a_matrices[idx]
        b = b_matrices[idx]
        q = q_matrices[idx]
        r = r_matrices[idx]

        # Derivation of this in the LQR Backward Recursion section of the following page:
        # https://confluence.ci.motional.com/confluence/x/VaOdCg
        k = -np.linalg.pinv(r + b.T @ p_current @ b) @ b.T @ p_current @ a
        p_current = q + k.T @ r @ k + (a + b @ k).T @ p_current @ (a + b @ k)

        k_matrices.append(k)
        p_matrices.append(p_current)

    # Reverse so we get ordering (K_0, P_0), (K_1, P_1), ... , (K_{N-1}, P_{N-1}).
    k_matrices.reverse()
    p_matrices.reverse()

    return np.array(k_matrices), np.array(p_matrices)


class ILQRSolver:
    """iLQR solver implementation, see module docstring for details."""

    def __init__(
        self,
        solver_params: ILQRSolverParameters,
        warm_start_params: ILQRWarmStartParameters,
    ) -> None:
        """
        Initialize solver parameters.
        :param solver_params: Contains solver parameters for iLQR.
        :param warm_start_params: Contains warm start parameters for iLQR.
        """
        self._solver_params = solver_params
        self._warm_start_params = warm_start_params

        self._n_states = 5  # state dimension
        self._n_inputs = 2  # input dimension

        q_diagonal_entries = self._solver_params.q_diagonal_entries
        assert len(q_diagonal_entries) == self._n_states, f"Q matrix should have diagonal length {self._n_states}."
        self._q: DoubleMatrix = np.diag(q_diagonal_entries)  # original state cost matrix

        r_diagonal_entries = self._solver_params.r_diagonal_entries
        assert len(r_diagonal_entries) == self._n_inputs, f"R matrix should have diagonal length {self._n_inputs}."
        self._r: DoubleMatrix = np.diag(r_diagonal_entries)  # original input cost matrix

    def solve(self, current_state: DoubleMatrix, reference_trajectory: DoubleMatrix) -> DoubleMatrix:
        """
        Run the main iLQR loop used to compute optimal inputs to track the reference trajectory from the current state.
        :param current_state: The initial state from which we apply inputs, z_0.
        :param reference_trajectory: The state reference we'd like to track, inclusive of the initial timestep,
                                     z_{r,k} for k in {0, ..., N}.
        :return: The optimal input sequence after iLQR converges or hits max iterations.
        """
        # Check that state parameter has the right shape.
        assert current_state.shape == (self._n_states,), "Incorrect state shape."

        # Check that reference trajectory parameter has the right shape.
        assert len(reference_trajectory.shape) == 2, "Reference trajectory should be a M x N matrix."
        reference_trajectory_length, reference_trajectory_state_dimension = reference_trajectory.shape
        assert (
            reference_trajectory_length > 1
        ), "The reference trajectory should be at least two timesteps (z_{r,0}, z_{r,1}) long."
        assert (
            reference_trajectory_state_dimension == self._n_states
        ), "The reference trajectory should have a matching state dimension."

        input_trajectory = self._input_warm_start(current_state, reference_trajectory)

        # Main iLQR Loop.
        for _ in range(self._solver_params.max_ilqr_iterations):
            # Run dynamics on the current input_trajectory iterate to get linearization states and Jacobians.
            state_trajectory, state_jacobian_trajectory, input_jacobian_trajectory = self._run_forward_dynamics(
                current_state, input_trajectory
            )

            # Lift the state and input according to the perturbed linear time varying (LTV) system dynamics.
            (
                state_jacobian_matrix_augmented_trajectory,
                input_jacobian_matrix_augmented_trajectory,
            ) = self._linear_augmented_dynamics_matrices(state_jacobian_trajectory, input_jacobian_trajectory)

            # Generate the cost matrices for the perturbed LTV system with trust region penalty.
            (
                state_cost_matrix_augmented_trajectory,
                input_cost_matrix_augmented_trajectory,
                state_cost_matrix_augmented_terminal,
            ) = self._quadratic_augmented_cost_matrices(input_trajectory, state_trajectory, reference_trajectory)

            # Use dynamic programming/backward recursion to compute the LQR state feedback gain matrices.
            # These gain matrices are applicable for the perturbed LTV system.
            state_feedback_matrix_augmented_trajectory, _ = _run_lqr_backward_recursion(
                a_matrices=state_jacobian_matrix_augmented_trajectory,
                b_matrices=input_jacobian_matrix_augmented_trajectory,
                q_matrices=state_cost_matrix_augmented_trajectory,
                r_matrices=input_cost_matrix_augmented_trajectory,
                q_terminal=state_cost_matrix_augmented_terminal,
            )

            # Determine locally optimal inputs by applying the feedback policy based on the LQR state feedback gain
            # matrices from the previous step.
            input_trajectory_next = self._update_input_sequence_with_feedback_policy(
                input_trajectory, state_trajectory, state_feedback_matrix_augmented_trajectory
            )

            # Check for convergence and terminate early if so.  Else update the input_trajectory iterate and continue.
            input_trajectory_norm_difference = np.linalg.norm(input_trajectory_next - input_trajectory)
            if input_trajectory_norm_difference < self._solver_params.convergence_threshold:
                break

            input_trajectory = input_trajectory_next

        return input_trajectory

    ####################################################################################################################
    # Helper methods.
    ####################################################################################################################

    def _clip_inputs(self, inputs: DoubleMatrix) -> DoubleMatrix:
        """
        Used to clip control inputs within constraints.
        :param: inputs: The control inputs with shape (2,) or control inputs array with shape (N,2) to clip.
        :return: Clipped version of the control inputs, unmodified if already within constraints.
        """
        assert inputs.ndim > 0, "Input should not be a scalar."
        assert inputs.ndim <= 2, "Input should be a vector or a matrix."

        inputs_clipped: DoubleMatrix = np.copy(inputs)

        max_acceleration = self._solver_params.max_acceleration
        max_steering_angle_rate = self._solver_params.max_steering_angle_rate

        if inputs_clipped.ndim == 1:
            assert inputs_clipped.shape == (self._n_inputs,), f"The expected input dimension is {self._n_inputs}."
            inputs_clipped[0] = np.clip(inputs_clipped[0], -max_acceleration, max_acceleration)
            inputs_clipped[1] = np.clip(inputs_clipped[1], -max_steering_angle_rate, max_steering_angle_rate)
        else:
            assert inputs_clipped.shape[1] == self._n_inputs, f"The expected input dimension is {self._n_inputs}."
            inputs_clipped[:, 0] = np.clip(inputs_clipped[:, 0], -max_acceleration, max_acceleration)
            inputs_clipped[:, 1] = np.clip(inputs_clipped[:, 1], -max_steering_angle_rate, max_steering_angle_rate)

        return inputs_clipped

    def _input_warm_start(self, current_state: DoubleMatrix, reference_trajectory: DoubleMatrix) -> DoubleMatrix:
        """
        Given a reference trajectory, we generate the warm start (initial guess) by inferring the inputs applied based
        on poses in the reference trajectory.
        :param current_state: The initial state from which we apply inputs.
        :param reference_trajectory: The reference trajectory we are trying to follow.
        :return: The warm start input trajectory estimated from poses.
        """
        reference_states_completed, reference_inputs_completed = complete_kinematic_state_and_inputs_from_poses(
            discretization_time=self._solver_params.discretization_time,
            wheel_base=self._solver_params.wheelbase,
            poses=reference_trajectory[:, :3],
            jerk_penalty=self._warm_start_params.jerk_penalty_warm_start_fit,
            curvature_rate_penalty=self._warm_start_params.curvature_rate_penalty_warm_start_fit,
        )

        # Now reference_inputs_completed can be thought of as feedforward inputs.
        # This works if current_state = reference_trajectory_completed[0,:] - i.e. no initial tracking error.
        # We add feedback input terms for the first control input only to account for nonzero initial tracking error.
        x_current, y_current, heading_current, velocity_current, steering_angle_current = current_state
        (
            x_reference,
            y_reference,
            heading_reference,
            velocity_reference,
            steering_angle_reference,
        ) = reference_states_completed[0, :]

        acceleration_feedback = -self._warm_start_params.k_velocity_error_feedback * (
            velocity_current - velocity_reference
        )

        steering_angle_feedback = compute_steering_angle_feedback(
            pose_reference=np.array([x_reference, y_reference, heading_reference]),
            pose_current=np.array([x_current, y_current, heading_current]),
            lookahead_distance=self._warm_start_params.lookahead_distance_lateral_error,
            k_lateral_error=self._warm_start_params.k_lateral_error,
        )
        steering_angle_desired = steering_angle_feedback + steering_angle_reference
        steering_rate_feedback = -self._warm_start_params.k_steering_angle_error_feedback * (
            steering_angle_current - steering_angle_desired
        )

        reference_inputs_completed[0, 0] += acceleration_feedback
        reference_inputs_completed[0, 1] += steering_rate_feedback

        return self._clip_inputs(reference_inputs_completed)

    def _update_input_sequence_with_feedback_policy(
        self,
        input_linearization_trajectory: DoubleMatrix,
        state_linearization_trajectory: DoubleMatrix,
        state_feedback_matrix_augmented_trajectory: DoubleMatrix,
    ) -> DoubleMatrix:
        """
        Given a linearization state and input trajectory and LQR feedback gain matrices,
        compute the next set of inputs after applying locally optimal perturbations.
        :param input_linearization_trajectory: The input trajectory about which we linearized, from timestamp 0 to N-1.
        :param state_linearization_trajectory: The corresponding state trajectory about which we linearized, from timestamp 0 to N.
        :param state_feedback_matrix_augmented_trajectory: The feedback gain matrices to determine input perturbations, from timestamp 0 to N-1.
        :return: The updated input sequence after applying perturbations, matching input_linearization_trajectory in shape.
        """
        assert len(input_linearization_trajectory) == len(
            state_feedback_matrix_augmented_trajectory
        ), "Input linearization trajectory and feedback policy matrices should have the same length."
        assert len(input_linearization_trajectory) + 1 == len(
            state_linearization_trajectory
        ), "The state linearization trajectory should be 1 longer than the input linearization trajectory."
        assert (
            input_linearization_trajectory.shape[1] == self._n_inputs
        ), f"The input linearization trajectory should have input dimension {self._n_inputs}."
        assert (
            state_linearization_trajectory.shape[1] == self._n_states
        ), f"The state linearization trajectory should have state dimension {self._n_states}."
        assert state_feedback_matrix_augmented_trajectory.shape[1:] == (
            self._n_inputs + 1,
            self._n_states + 1,
        ), f"The feedback gain matrices should have shape {(self._n_inputs+1, self._n_states+1)}"

        # Trajectory of state perturbations while applying feedback policy.
        # Starts with zero as the initial states match exactly, only later states might vary.
        delta_state_trajectory = [np.zeros(self._n_states)]

        # This is the updated input trajectory we will return after applying the input perturbations.
        input_next_trajectory = []

        # Used to extract input perturbation after applying feedback gain matrix.
        input_mask_matrix: DoubleMatrix = np.block([np.eye(self._n_inputs), np.zeros((self._n_inputs, 1))])

        zip_object = zip(
            input_linearization_trajectory,
            state_linearization_trajectory[:-1],
            state_linearization_trajectory[1:],
            state_feedback_matrix_augmented_trajectory,
        )

        for (input_lin, state_lin, state_lin_next, state_feedback_matrix) in zip_object:
            # Compute locally optimal input perturbation.
            delta_state = delta_state_trajectory[-1]
            delta_input = input_mask_matrix @ state_feedback_matrix @ np.block([delta_state, 1])

            # Apply input perturbation within input constraints.
            input_perturbed = self._clip_inputs(input_lin + delta_input)

            # Apply state perturbation.
            state_perturbed = state_lin + delta_state
            state_perturbed[2] = principal_value(state_perturbed[2])

            # Run dynamics with perturbed state/inputs to get next state.
            state_perturbed_next, _, _ = self._dynamics_and_jacobian(state_perturbed, input_perturbed)

            # Compute next state perturbation given next state.
            delta_state_next = state_perturbed_next - state_lin_next
            delta_state_next[2] = principal_value(delta_state_next[2])

            delta_state_trajectory.append(delta_state_next)
            input_next_trajectory.append(input_perturbed)

        return np.array(input_next_trajectory)

    ####################################################################################################################
    # Dynamics and Jacobian.
    ####################################################################################################################

    def _run_forward_dynamics(
        self, current_state: DoubleMatrix, input_trajectory: DoubleMatrix
    ) -> Tuple[DoubleMatrix, DoubleMatrix, DoubleMatrix]:
        """
        Compute states and corresponding state/input jacobian matrices using forward dynamics.
        :param current_state: The initial state from which we apply inputs.
        :param input_trajectory: The input trajectory applied to the model - NOT checked for constraint satisfaction.
        :return: The state, state jacobian, and input jacobian trajectories.
        """
        state_trajectory = [current_state]  # Result includes current_state, z_0.
        state_jacobian_trajectory = []
        input_jacobian_trajectory = []

        for u in input_trajectory:
            state_next, state_jacobian, input_jacobian = self._dynamics_and_jacobian(state_trajectory[-1], u)

            state_trajectory.append(state_next)
            state_jacobian_trajectory.append(state_jacobian)
            input_jacobian_trajectory.append(input_jacobian)

        return np.array(state_trajectory), np.array(state_jacobian_trajectory), np.array(input_jacobian_trajectory)

    def _dynamics_and_jacobian(
        self, current_state: DoubleMatrix, current_input: DoubleMatrix
    ) -> Tuple[DoubleMatrix, DoubleMatrix, DoubleMatrix]:
        """
        Propagates the state forward by one step and computes the corresponding state and input Jacobian matrices.
        :param current_state: The current state z.
        :param current_input: The applied input u.
        :return: The next state z' reached by applying u from z and state Jacobian (df/dz) and input Jacobian (df/du).
        """
        x, y, heading, velocity, steering_angle = current_state
        acceleration, steering_rate = current_input

        # Check steering angle is in expected range for valid Jacobian matrices.
        assert (
            np.abs(steering_angle) < np.pi / 2.0
        ), f"The steering angle {steering_angle} is outside expected limits.  There is a singularity at delta = np.pi/2."

        # Euler integration of bicycle model.
        discretization_time = self._solver_params.discretization_time
        wheelbase = self._solver_params.wheelbase

        next_state: DoubleMatrix = np.copy(current_state)
        next_state[0] += velocity * np.cos(heading) * discretization_time
        next_state[1] += velocity * np.sin(heading) * discretization_time
        next_state[2] += velocity * np.tan(steering_angle) / wheelbase * discretization_time
        next_state[3] += acceleration * discretization_time
        next_state[4] += steering_rate * discretization_time

        # Constrain heading angle to lie within +/- pi.
        next_state[2] = principal_value(next_state[2])

        # Now we construct and populate the state and input Jacobians.
        state_jacobian: DoubleMatrix = np.eye(self._n_states, dtype=np.float64)
        input_jacobian: DoubleMatrix = np.zeros((self._n_states, self._n_inputs), dtype=np.float64)

        # Set a nonzero velocity to handle issues when linearizing at (near) zero velocity.
        # This helps e.g. when the vehicle is stopped with zero steering angle and needs to accelerate/turn.
        # Without this, the A matrix will indicate steering has no impact on heading due to Euler discretization.
        # There will be a rank drop in the controllability matrix, so the discrete-time algebraic Riccati equation
        # may not have a solution (uncontrollable subspace) or it may not be unique.
        min_velocity_linearization = self._solver_params.min_velocity_linearization
        if -min_velocity_linearization <= velocity and velocity <= min_velocity_linearization:
            sign_velocity = 1.0 if velocity >= 0.0 else -1.0
            velocity = sign_velocity * min_velocity_linearization

        state_jacobian[0, 2] = -velocity * np.sin(heading) * discretization_time
        state_jacobian[0, 3] = np.cos(heading) * discretization_time

        state_jacobian[1, 2] = velocity * np.cos(heading) * discretization_time
        state_jacobian[1, 3] = np.sin(heading) * discretization_time

        state_jacobian[2, 3] = np.tan(steering_angle) / wheelbase * discretization_time
        state_jacobian[2, 4] = velocity * discretization_time / (wheelbase * np.cos(steering_angle) ** 2)

        input_jacobian[3, 0] = discretization_time
        input_jacobian[4, 1] = discretization_time

        return next_state, state_jacobian, input_jacobian

    ####################################################################################################################
    # Augmented Dynamics and Cost Implementation.
    ####################################################################################################################

    def _linear_augmented_dynamics_matrices(
        self, state_jacobian_trajectory: DoubleMatrix, input_jacobian_trajectory: DoubleMatrix
    ) -> Tuple[DoubleMatrix, DoubleMatrix]:
        """
        Returns the matrices for the linear time varying system describing the perturbation "delta"-dynamics.
        We mean here how small deviations in state (delta_state) and input (delta_input) about the linearization trajectory
        change the resultant perturbed trajectory.

        The input is the original Jacobians (A_k, B_k) about the linearized trajectory.
        This function returns the augmented system matrices as described below:
        | delta_state,{k+1} | = | A_k 0 | | delta_state,k | + |B_k  0 | | delta_input,k |
        |       1           |   |  0  1 | |     1         |   | 0   0 | |     1         |

        :param state_jacobian_trajectory: a trajectory of state Jacobian matrices evaluated at each timestep k
        :param input_jacobian_trajectory: a trajectory of input Jacobian matrices evaluated at each timestep k
        :return: A trajectory of augmented state and input matrices matching the augmented linear time varying system.
        """
        state_jacobian_matrix_augmented_trajectory: List[DoubleMatrix] = []
        input_jacobian_matrix_augmented_trajectory: List[DoubleMatrix] = []

        for state_jacobian, input_jacobian in zip(state_jacobian_trajectory, input_jacobian_trajectory):
            state_jacobian_augmented: DoubleMatrix = np.zeros(
                (self._n_states + 1, self._n_states + 1), dtype=np.float64
            )
            state_jacobian_augmented[: self._n_states, : self._n_states] = state_jacobian
            state_jacobian_augmented[-1, -1] = 1.0

            input_jacobian_augmented: DoubleMatrix = np.zeros(
                (self._n_states + 1, self._n_inputs + 1), dtype=np.float64
            )
            input_jacobian_augmented[: self._n_states, : self._n_inputs] = input_jacobian

            state_jacobian_matrix_augmented_trajectory.append(state_jacobian_augmented)
            input_jacobian_matrix_augmented_trajectory.append(input_jacobian_augmented)

        return np.array(state_jacobian_matrix_augmented_trajectory), np.array(
            input_jacobian_matrix_augmented_trajectory
        )

    def _quadratic_augmented_cost_matrices(
        self,
        input_linearization_trajectory: DoubleMatrix,
        state_linearization_trajectory: DoubleMatrix,
        reference_trajectory: DoubleMatrix,
    ) -> Tuple[DoubleMatrix, DoubleMatrix, DoubleMatrix]:
        """
        Returns the matrices capturing the quadratic cost matrices for the augmented linear time varying system
        with "delta"-dynamics.  We also incorporate a trust region penalty to avoid large delta deviations from the
        prior iterate (input_linearization_trajectory, state_linearization_trajectory).
        :param input_linearization_trajectory: Input trajectory about which we linearize, bar{u}_k for k = 0, ..., N-1.
        :param state_linearization_trajectory: State trajectory about which we linearize, bar{z}_k for k = 0, ..., N.
        :param reference_trajectory: Reference trajectory we are trying to track, z_{r,k} for k = 0, ..., N.
        :return: A trajectory of augmented Q (state) and R (input) stage cost matrices.
                In addition, we provide the terminal cost matrix separately.
        """
        assert (
            len(input_linearization_trajectory) == len(state_linearization_trajectory) - 1
        ), "The input trajectory should be one step shorter than the state trajectory with terminal state."
        assert len(state_linearization_trajectory) == len(
            reference_trajectory
        ), "The state and reference trajectory should have the same length."

        state_cost_matrix_augmented_trajectory: List[DoubleMatrix] = []
        input_cost_matrix_augmented_trajectory: List[DoubleMatrix] = []

        # Trust region penalty term for states, constant throughout the time horizon.
        q_trust_region = np.zeros((self._n_states + 1, self._n_states + 1))
        q_trust_region[: self._n_states, : self._n_states] = np.eye(self._n_states)

        # Trust region penalty term for inputs, constant throughout the time horizon.
        r_trust_region = np.zeros((self._n_inputs + 1, self._n_inputs + 1))
        r_trust_region[: self._n_inputs, : self._n_inputs] = np.eye(self._n_inputs)

        # This is the tracking error between the linearization and reference trajectories.
        state_error_trajectory = state_linearization_trajectory - reference_trajectory
        state_error_trajectory[:, 2] = principal_value(state_error_trajectory[:, 2])

        alpha_trust_region = self._solver_params.alpha_trust_region

        for state_idx, (input_lin, state_error) in enumerate(
            zip(input_linearization_trajectory, state_error_trajectory)
        ):
            q_cost: DoubleMatrix = np.zeros((self._n_states + 1, self._n_states + 1))

            if state_idx > 0:
                # Note: we do not penalize the first state error as our choice of control inputs cannot change this.
                off_diagonal_q_cost = self._q @ state_error

                q_cost = np.block(
                    [
                        [self._q, np.expand_dims(off_diagonal_q_cost, axis=-1)],
                        [off_diagonal_q_cost, state_error.T @ self._q @ state_error],
                    ]
                )

            state_cost_matrix_augmented = (1 - alpha_trust_region) * q_cost + alpha_trust_region * q_trust_region

            off_diagonal_r_cost = self._r @ input_lin
            r_cost: DoubleMatrix = np.block(
                [
                    [self._r, np.expand_dims(off_diagonal_r_cost, axis=-1)],
                    [off_diagonal_r_cost, input_lin.T @ self._r @ input_lin],
                ]
            )

            input_cost_matrix_augmented = (1 - alpha_trust_region) * r_cost + alpha_trust_region * r_trust_region

            state_cost_matrix_augmented_trajectory.append(state_cost_matrix_augmented)
            input_cost_matrix_augmented_trajectory.append(input_cost_matrix_augmented)

        state_error_terminal = state_error_trajectory[-1]

        off_diagonal_q_cost = self._q @ state_error_terminal
        q_cost_terminal: DoubleMatrix = np.block(
            [
                [self._q, np.expand_dims(off_diagonal_q_cost, axis=-1)],
                [off_diagonal_q_cost, state_error_terminal.T @ self._q @ state_error_terminal],
            ]
        )

        state_cost_matrix_augmented_terminal = (
            1 - alpha_trust_region
        ) * q_cost_terminal + alpha_trust_region * q_trust_region

        return (
            np.array(state_cost_matrix_augmented_trajectory),
            np.array(input_cost_matrix_augmented_trajectory),
            state_cost_matrix_augmented_terminal,
        )
