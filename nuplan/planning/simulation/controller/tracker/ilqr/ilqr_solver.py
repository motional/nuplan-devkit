"""
This provides an implementation of the iterative linear quadratic regulator (iLQR) algorithm for trajectory tracking.
It is specialized to the case with a discrete-time kinematic bicycle model and a quadratic trajectory tracking cost.

Original (Nonlinear) Discrete Time System:
    z_k = [x_k, y_k, theta_k, v_k, delta_k]
    u_k = [a_k, phi_k]

    x_{k+1}     = x_k     + v_k * cos(theta_k) * dt
    y_{k+1}     = y_k     + v_k * sin(theta_k) * dt
    theta_{k+1} = theta_k + v_k * tan(delta_k) / L * dt
    v_{k+1}     = v_k     + a_k * dt
    delta_{k+1} = delta_k + phi_k * dt

    where (x_k, y_k, theta_k) is the pose at timestep k with time discretization dt,
    v_k and a_k are velocity and acceleration,
    delta_k and phi_k are steering angle and steering angle rate,
    and L is the vehicle wheelbase.

Quadratic Tracking Cost:
    J = sum_{k=0}^{N-1} ||u_k||_2^{R_k} +
        sum_{k=0}^N ||z_k - z_{ref,k}||_2^{Q_k}
For simplicity, we opt to use constant input cost matrices R_k = R and constant state cost matrices Q_k = Q.

There are multiple improvements that can be done for this implementation, but omitted for simplicity of the code.
Some of these include:
  * Handle constraints directly in the optimization (e.g. log-barrier / penalty method with quadratic cost estimate).
  * Line search in the input policy update (feedforward term) to determine a good gradient step size.

References Used: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec5-LQR.pdf and
                 https://www.cs.cmu.edu/~rsalakhu/10703/Lectures/Lecture_trajectoryoptimization.pdf
"""

import time
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

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
    """Parameters related to the solver implementation."""

    discretization_time: float  # [s] Time discretization used for integration.

    # Cost weights for state [x, y, heading, velocity, steering angle] and input variables [acceleration, steering rate].
    state_cost_diagonal_entries: List[float]
    input_cost_diagonal_entries: List[float]

    # Trust region cost weights for state and input variables.  Helps keep linearization error per update step bounded.
    state_trust_region_entries: List[float]
    input_trust_region_entries: List[float]

    # Parameters related to solver runtime / solution sub-optimality.
    max_ilqr_iterations: int  # Maximum number of iterations to run iLQR before timeout.
    convergence_threshold: float  # Threshold for delta inputs below which we can terminate iLQR early.
    max_solve_time: Optional[
        float
    ]  # [s] If defined, sets a maximum time to run a solve call of iLQR before terminating.

    # Constraints for underlying dynamics model.
    max_acceleration: float  # [m/s^2] Absolute value threshold on acceleration input.
    max_steering_angle: float  # [rad] Absolute value threshold on steering angle state.
    max_steering_angle_rate: float  # [rad/s] Absolute value threshold on steering rate input.

    # Parameters for dynamics / linearization.
    min_velocity_linearization: float  # [m/s] Absolute value threshold below which linearization velocity is modified.
    wheelbase: float = get_pacifica_parameters().wheel_base  # [m] Wheelbase length parameter for the vehicle.

    def __post_init__(self) -> None:
        """Ensure entries lie in expected bounds and initialize wheelbase."""
        for entry in [
            "discretization_time",
            "max_ilqr_iterations",
            "convergence_threshold",
            "max_acceleration",
            "max_steering_angle",
            "max_steering_angle_rate",
            "min_velocity_linearization",
            "wheelbase",
        ]:
            assert getattr(self, entry) > 0.0, f"Field {entry} should be positive."

        assert self.max_steering_angle < np.pi / 2.0, "Max steering angle should be less than 90 degrees."

        if isinstance(self.max_solve_time, float):
            assert self.max_solve_time > 0.0, "The specified max solve time should be positive."

        assert np.all([x >= 0 for x in self.state_cost_diagonal_entries]), "Q matrix must be positive semidefinite."
        assert np.all([x > 0 for x in self.input_cost_diagonal_entries]), "R matrix must be positive definite."

        assert np.all(
            [x > 0 for x in self.state_trust_region_entries]
        ), "State trust region cost matrix must be positive definite."
        assert np.all(
            [x > 0 for x in self.input_trust_region_entries]
        ), "Input trust region cost matrix must be positive definite."


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


@dataclass(frozen=True)
class ILQRIterate:
    """Contains state, input, and associated Jacobian trajectories needed to perform an update step of iLQR."""

    state_trajectory: DoubleMatrix
    input_trajectory: DoubleMatrix
    state_jacobian_trajectory: DoubleMatrix
    input_jacobian_trajectory: DoubleMatrix

    def __post_init__(self) -> None:
        """Check consistency of dimension across trajectory elements."""
        assert len(self.state_trajectory.shape) == 2, "Expect state trajectory to be a 2D matrix."
        state_trajectory_length, state_dim = self.state_trajectory.shape

        assert len(self.input_trajectory.shape) == 2, "Expect input trajectory to be a 2D matrix."
        input_trajectory_length, input_dim = self.input_trajectory.shape

        assert (
            input_trajectory_length == state_trajectory_length - 1
        ), "State trajectory should be 1 longer than the input trajectory."
        assert self.state_jacobian_trajectory.shape == (input_trajectory_length, state_dim, state_dim)
        assert self.input_jacobian_trajectory.shape == (input_trajectory_length, state_dim, input_dim)

        for field in fields(self):
            # Make sure that we have no nan entries in our trajectory rollout prior to operating on this.
            assert ~np.any(np.isnan(getattr(self, field.name))), f"{field.name} has unexpected nan values."


@dataclass(frozen=True)
class ILQRInputPolicy:
    """Contains parameters for the perturbation input policy computed after performing LQR."""

    state_feedback_matrices: DoubleMatrix
    feedforward_inputs: DoubleMatrix

    def __post__init__(self) -> None:
        """Check shape of policy parameters."""
        assert (
            len(self.state_feedback_matrices.shape) == 3
        ), "Expected state_feedback_matrices to have shape (n_horizon, n_inputs, n_states)"

        assert (
            len(self.feedforward_inputs.shape) == 2
        ), "Expected feedforward inputs to have shape (n_horizon, n_inputs)."

        assert (
            self.feedforward_inputs.shape == self.state_feedback_matrices.shape[:2]
        ), "Inconsistent horizon or input dimension between feedforward inputs and state feedback matrices."

        for field in fields(self):
            # Make sure that we have no nan entries in our policy parameters prior to using them.
            assert ~np.any(np.isnan(getattr(self, field.name))), f"{field.name} has unexpected nan values."


@dataclass(frozen=True)
class ILQRSolution:
    """Contains the iLQR solution with associated cost for consumption by the solver's client."""

    state_trajectory: DoubleMatrix
    input_trajectory: DoubleMatrix

    tracking_cost: float

    def __post_init__(self) -> None:
        """Check consistency of dimension across trajectory elements and nonnegative cost."""
        assert len(self.state_trajectory.shape) == 2, "Expect state trajectory to be a 2D matrix."
        state_trajectory_length, _ = self.state_trajectory.shape

        assert len(self.input_trajectory.shape) == 2, "Expect input trajectory to be a 2D matrix."
        input_trajectory_length, _ = self.input_trajectory.shape

        assert (
            input_trajectory_length == state_trajectory_length - 1
        ), "State trajectory should be 1 longer than the input trajectory."

        assert self.tracking_cost >= 0.0, "Expect the tracking cost to be nonnegative."


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

        state_cost_diagonal_entries = self._solver_params.state_cost_diagonal_entries
        assert (
            len(state_cost_diagonal_entries) == self._n_states
        ), f"State cost matrix should have diagonal length {self._n_states}."
        self._state_cost_matrix: DoubleMatrix = np.diag(state_cost_diagonal_entries)

        input_cost_diagonal_entries = self._solver_params.input_cost_diagonal_entries
        assert (
            len(input_cost_diagonal_entries) == self._n_inputs
        ), f"Input cost matrix should have diagonal length {self._n_inputs}."
        self._input_cost_matrix: DoubleMatrix = np.diag(input_cost_diagonal_entries)

        state_trust_region_entries = self._solver_params.state_trust_region_entries
        assert (
            len(state_trust_region_entries) == self._n_states
        ), f"State trust region cost matrix should have diagonal length {self._n_states}."
        self._state_trust_region_cost_matrix: DoubleMatrix = np.diag(state_trust_region_entries)

        input_trust_region_entries = self._solver_params.input_trust_region_entries
        assert (
            len(input_trust_region_entries) == self._n_inputs
        ), f"Input trust region cost matrix should have diagonal length {self._n_inputs}."
        self._input_trust_region_cost_matrix: DoubleMatrix = np.diag(input_trust_region_entries)

        max_acceleration = self._solver_params.max_acceleration
        max_steering_angle_rate = self._solver_params.max_steering_angle_rate

        # Define input clip limits once to avoid recomputation in _clip_inputs.
        self._input_clip_min = (-max_acceleration, -max_steering_angle_rate)
        self._input_clip_max = (max_acceleration, max_steering_angle_rate)

    def solve(self, current_state: DoubleMatrix, reference_trajectory: DoubleMatrix) -> List[ILQRSolution]:
        """
        Run the main iLQR loop used to try to find (locally) optimal inputs to track the reference trajectory.
        :param current_state: The initial state from which we apply inputs, z_0.
        :param reference_trajectory: The state reference we'd like to track, inclusive of the initial timestep,
                                     z_{r,k} for k in {0, ..., N}.
        :return: A list of solution iterates after running the iLQR algorithm where the index is the iteration number.
        """
        # Check that state parameter has the right shape.
        assert current_state.shape == (self._n_states,), "Incorrect state shape."

        # Check that reference trajectory parameter has the right shape.
        assert len(reference_trajectory.shape) == 2, "Reference trajectory should be a 2D matrix."
        reference_trajectory_length, reference_trajectory_state_dimension = reference_trajectory.shape
        assert reference_trajectory_length > 1, "The reference trajectory should be at least two timesteps long."
        assert (
            reference_trajectory_state_dimension == self._n_states
        ), "The reference trajectory should have a matching state dimension."

        # List of ILQRSolution results where the index corresponds to the iteration of iLQR.
        solution_list: List[ILQRSolution] = []

        # Get warm start input and state trajectory, as well as associated Jacobians.
        current_iterate = self._input_warm_start(current_state, reference_trajectory)

        # Main iLQR Loop.
        solve_start_time = time.perf_counter()
        for _ in range(self._solver_params.max_ilqr_iterations):
            # Determine the cost and store the associated solution object.
            tracking_cost = self._compute_tracking_cost(
                iterate=current_iterate,
                reference_trajectory=reference_trajectory,
            )
            solution_list.append(
                ILQRSolution(
                    input_trajectory=current_iterate.input_trajectory,
                    state_trajectory=current_iterate.state_trajectory,
                    tracking_cost=tracking_cost,
                )
            )

            # Determine the LQR optimal perturbations to apply.
            lqr_input_policy = self._run_lqr_backward_recursion(
                current_iterate=current_iterate,
                reference_trajectory=reference_trajectory,
            )

            # Apply the optimal perturbations to generate the next input trajectory iterate.
            input_trajectory_next = self._update_inputs_with_policy(
                current_iterate=current_iterate,
                lqr_input_policy=lqr_input_policy,
            )

            # Check for convergence/timeout and terminate early if so.
            # Else update the input_trajectory iterate and continue.
            input_trajectory_norm_difference = np.linalg.norm(input_trajectory_next - current_iterate.input_trajectory)

            current_iterate = self._run_forward_dynamics(current_state, input_trajectory_next)

            if input_trajectory_norm_difference < self._solver_params.convergence_threshold:
                break

            elapsed_time = time.perf_counter() - solve_start_time
            if (
                isinstance(self._solver_params.max_solve_time, float)
                and elapsed_time >= self._solver_params.max_solve_time
            ):
                break

        # Store the final iterate in the solution_dict.
        tracking_cost = self._compute_tracking_cost(
            iterate=current_iterate,
            reference_trajectory=reference_trajectory,
        )
        solution_list.append(
            ILQRSolution(
                input_trajectory=current_iterate.input_trajectory,
                state_trajectory=current_iterate.state_trajectory,
                tracking_cost=tracking_cost,
            )
        )

        return solution_list

    ####################################################################################################################
    # Helper methods.
    ####################################################################################################################

    def _compute_tracking_cost(self, iterate: ILQRIterate, reference_trajectory: DoubleMatrix) -> float:
        """
        Compute the trajectory tracking cost given a candidate solution.
        :param iterate: Contains the candidate state and input trajectory to evaluate.
        :param reference_trajectory: The desired state reference trajectory with same length as state_trajectory.
        :return: The tracking cost of the candidate state/input trajectory.
        """
        input_trajectory = iterate.input_trajectory
        state_trajectory = iterate.state_trajectory

        assert len(state_trajectory) == len(
            reference_trajectory
        ), "The state and reference trajectory should have the same length."

        error_state_trajectory = state_trajectory - reference_trajectory
        error_state_trajectory[:, 2] = principal_value(error_state_trajectory[:, 2])

        cost = np.sum([u.T @ self._input_cost_matrix @ u for u in input_trajectory]) + np.sum(
            [e.T @ self._state_cost_matrix @ e for e in error_state_trajectory]
        )

        return float(cost)

    def _clip_inputs(self, inputs: DoubleMatrix) -> DoubleMatrix:
        """
        Used to clip control inputs within constraints.
        :param: inputs: The control inputs with shape (self._n_inputs,) to clip.
        :return: Clipped version of the control inputs, unmodified if already within constraints.
        """
        assert inputs.shape == (self._n_inputs,), f"The inputs should be a 1D vector with {self._n_inputs} elements."

        return np.clip(inputs, self._input_clip_min, self._input_clip_max)  # type: ignore

    def _clip_steering_angle(self, steering_angle: float) -> float:
        """
        Used to clip the steering angle state within bounds.
        :param steering_angle: [rad] A steering angle (scalar) to clip.
        :return: [rad] The clipped steering angle.
        """
        steering_angle_sign = 1.0 if steering_angle >= 0 else -1.0
        steering_angle = steering_angle_sign * min(abs(steering_angle), self._solver_params.max_steering_angle)
        return steering_angle

    def _input_warm_start(self, current_state: DoubleMatrix, reference_trajectory: DoubleMatrix) -> ILQRIterate:
        """
        Given a reference trajectory, we generate the warm start (initial guess) by inferring the inputs applied based
        on poses in the reference trajectory.
        :param current_state: The initial state from which we apply inputs.
        :param reference_trajectory: The reference trajectory we are trying to follow.
        :return: The warm start iterate from which to start iLQR.
        """
        reference_states_completed, reference_inputs_completed = complete_kinematic_state_and_inputs_from_poses(
            discretization_time=self._solver_params.discretization_time,
            wheel_base=self._solver_params.wheelbase,
            poses=reference_trajectory[:, :3],
            jerk_penalty=self._warm_start_params.jerk_penalty_warm_start_fit,
            curvature_rate_penalty=self._warm_start_params.curvature_rate_penalty_warm_start_fit,
        )

        # We could just stop here and apply reference_inputs_completed (assuming it satisfies constraints).
        # This could work if current_state = reference_states_completed[0,:] - i.e. no initial tracking error.
        # We add feedback input terms for the first control input only to account for nonzero initial tracking error.
        _, _, _, velocity_current, steering_angle_current = current_state
        _, _, _, velocity_reference, steering_angle_reference = reference_states_completed[0, :]

        acceleration_feedback = -self._warm_start_params.k_velocity_error_feedback * (
            velocity_current - velocity_reference
        )

        steering_angle_feedback = compute_steering_angle_feedback(
            pose_reference=current_state[:3],
            pose_current=reference_states_completed[0, :3],
            lookahead_distance=self._warm_start_params.lookahead_distance_lateral_error,
            k_lateral_error=self._warm_start_params.k_lateral_error,
        )
        steering_angle_desired = steering_angle_feedback + steering_angle_reference
        steering_rate_feedback = -self._warm_start_params.k_steering_angle_error_feedback * (
            steering_angle_current - steering_angle_desired
        )

        reference_inputs_completed[0, 0] += acceleration_feedback
        reference_inputs_completed[0, 1] += steering_rate_feedback

        # We rerun dynamics with constraints applied to make sure we have a feasible warm start for iLQR.
        return self._run_forward_dynamics(current_state, reference_inputs_completed)

    ####################################################################################################################
    # Dynamics and Jacobian.
    ####################################################################################################################

    def _run_forward_dynamics(self, current_state: DoubleMatrix, input_trajectory: DoubleMatrix) -> ILQRIterate:
        """
        Compute states and corresponding state/input Jacobian matrices using forward dynamics.
        We additionally return the input since the dynamics may modify the input to ensure constraint satisfaction.
        :param current_state: The initial state from which we apply inputs.  Must be feasible given constraints.
        :param input_trajectory: The input trajectory applied to the model.  May be modified to ensure feasibility.
        :return: A feasible iterate after applying dynamics with state/input trajectories and Jacobian matrices.
        """
        # Store rollout as a set of numpy arrays, initialized as np.nan to ensure we correctly fill them in.
        # The state trajectory includes the current_state, z_0, and is 1 element longer than the other arrays.
        # The final_input_trajectory captures the applied input for the dynamics model satisfying constraints.
        N = len(input_trajectory)
        state_trajectory = np.nan * np.ones((N + 1, self._n_states), dtype=np.float64)
        final_input_trajectory = np.nan * np.ones_like(input_trajectory, dtype=np.float64)
        state_jacobian_trajectory = np.nan * np.ones((N, self._n_states, self._n_states), dtype=np.float64)
        final_input_jacobian_trajectory = np.nan * np.ones((N, self._n_states, self._n_inputs), dtype=np.float64)

        state_trajectory[0] = current_state

        for idx_u, u in enumerate(input_trajectory):
            state_next, final_input, state_jacobian, final_input_jacobian = self._dynamics_and_jacobian(
                state_trajectory[idx_u], u
            )

            state_trajectory[idx_u + 1] = state_next
            final_input_trajectory[idx_u] = final_input
            state_jacobian_trajectory[idx_u] = state_jacobian
            final_input_jacobian_trajectory[idx_u] = final_input_jacobian

        iterate = ILQRIterate(
            state_trajectory=state_trajectory,  # type: ignore
            input_trajectory=final_input_trajectory,  # type: ignore
            state_jacobian_trajectory=state_jacobian_trajectory,  # type: ignore
            input_jacobian_trajectory=final_input_jacobian_trajectory,  # type: ignore
        )

        return iterate

    def _dynamics_and_jacobian(
        self, current_state: DoubleMatrix, current_input: DoubleMatrix
    ) -> Tuple[DoubleMatrix, DoubleMatrix, DoubleMatrix, DoubleMatrix]:
        """
        Propagates the state forward by one step and computes the corresponding state and input Jacobian matrices.
        We also impose all constraints here to ensure the current input and next state are always feasible.
        :param current_state: The current state z_k.
        :param current_input: The applied input u_k.
        :return: The next state z_{k+1}, (possibly modified) input u_k, and state (df/dz) and input (df/du) Jacobians.
        """
        x, y, heading, velocity, steering_angle = current_state

        # Check steering angle is in expected range for valid Jacobian matrices.
        assert (
            np.abs(steering_angle) < np.pi / 2.0
        ), f"The steering angle {steering_angle} is outside expected limits.  There is a singularity at delta = np.pi/2."

        # Input constraints: clip inputs within bounds and then use.
        current_input = self._clip_inputs(current_input)
        acceleration, steering_rate = current_input

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

        # State constraints: clip the steering_angle within bounds and update steering_rate accordingly.
        next_steering_angle = self._clip_steering_angle(next_state[4])
        applied_steering_rate = (next_steering_angle - steering_angle) / discretization_time
        next_state[4] = next_steering_angle
        current_input[1] = applied_steering_rate

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

        return next_state, current_input, state_jacobian, input_jacobian

    ####################################################################################################################
    # Core LQR implementation.
    ####################################################################################################################

    def _run_lqr_backward_recursion(
        self,
        current_iterate: ILQRIterate,
        reference_trajectory: DoubleMatrix,
    ) -> ILQRInputPolicy:
        """
        Computes the locally optimal affine state feedback policy by applying dynamic programming to linear perturbation
        dynamics about a specified linearization trajectory.  We include a trust region penalty as part of the cost.
        :param current_iterate: Contains all relevant linearization information needed to compute LQR policy.
        :param reference_trajectory: The desired state trajectory we are tracking.
        :return: An affine state feedback policy - state feedback matrices and feedforward inputs found using LQR.
        """
        state_trajectory = current_iterate.state_trajectory
        input_trajectory = current_iterate.input_trajectory
        state_jacobian_trajectory = current_iterate.state_jacobian_trajectory
        input_jacobian_trajectory = current_iterate.input_jacobian_trajectory

        # Check reference matches the expected shape.
        assert reference_trajectory.shape == state_trajectory.shape, "The reference trajectory has incorrect shape."

        # Compute nominal error trajectory.
        error_state_trajectory = state_trajectory - reference_trajectory
        error_state_trajectory[:, 2] = principal_value(error_state_trajectory[:, 2])

        # The value function has the form V_k(\Delta z_k) = \Delta z_k^T P_k \Delta z_k + 2 \rho_k^T \Delta z_k.
        # So p_current = P_k is related to the Hessian of the value function at the current timestep.
        # And rho_current = rho_k is part of the linear cost term in the value function at the current timestep.
        p_current = self._state_cost_matrix + self._state_trust_region_cost_matrix
        rho_current = self._state_cost_matrix @ error_state_trajectory[-1]

        # The optimal LQR policy has the form \Delta u_k^* = K_k \Delta z_k + \kappa_k
        # We refer to K_k as state_feedback_matrix and \kappa_k as feedforward input in the code below.
        N = len(input_trajectory)
        state_feedback_matrices = np.nan * np.ones((N, self._n_inputs, self._n_states), dtype=np.float64)
        feedforward_inputs = np.nan * np.ones((N, self._n_inputs), dtype=np.float64)

        for i in reversed(range(N)):
            A = state_jacobian_trajectory[i]
            B = input_jacobian_trajectory[i]
            u = input_trajectory[i]
            error = error_state_trajectory[i]

            # Compute the optimal input policy for this timestep.
            inverse_matrix_term = np.linalg.inv(
                self._input_cost_matrix + self._input_trust_region_cost_matrix + B.T @ p_current @ B
            )  # invertible since we checked input_cost / input_trust_region_cost are positive definite during creation.
            state_feedback_matrix = -inverse_matrix_term @ B.T @ p_current @ A
            feedforward_input = -inverse_matrix_term @ (self._input_cost_matrix @ u + B.T @ rho_current)

            # Compute the optimal value function for this timestep.
            a_closed_loop = A + B @ state_feedback_matrix

            p_prior = (
                self._state_cost_matrix
                + self._state_trust_region_cost_matrix
                + state_feedback_matrix.T @ self._input_cost_matrix @ state_feedback_matrix
                + state_feedback_matrix.T @ self._input_trust_region_cost_matrix @ state_feedback_matrix
                + a_closed_loop.T @ p_current @ a_closed_loop
            )

            rho_prior = (
                self._state_cost_matrix @ error
                + state_feedback_matrix.T @ self._input_cost_matrix @ (feedforward_input + u)
                + state_feedback_matrix.T @ self._input_trust_region_cost_matrix @ feedforward_input
                + a_closed_loop.T @ p_current @ B @ feedforward_input
                + a_closed_loop.T @ rho_current
            )

            p_current = p_prior
            rho_current = rho_prior

            state_feedback_matrices[i] = state_feedback_matrix
            feedforward_inputs[i] = feedforward_input

        lqr_input_policy = ILQRInputPolicy(
            state_feedback_matrices=state_feedback_matrices,  # type: ignore
            feedforward_inputs=feedforward_inputs,  # type: ignore
        )

        return lqr_input_policy

    def _update_inputs_with_policy(
        self,
        current_iterate: ILQRIterate,
        lqr_input_policy: ILQRInputPolicy,
    ) -> DoubleMatrix:
        """
        Used to update an iterate of iLQR by applying a perturbation input policy for local cost improvement.
        :param current_iterate: Contains the state and input trajectory about which we linearized.
        :param lqr_input_policy: Contains the LQR policy to apply.
        :return: The next input trajectory found by applying the LQR policy.
        """
        state_trajectory = current_iterate.state_trajectory
        input_trajectory = current_iterate.input_trajectory

        # Trajectory of state perturbations while applying feedback policy.
        # Starts with zero as the initial states match exactly, only later states might vary.
        delta_state_trajectory = np.nan * np.ones((len(input_trajectory) + 1, self._n_states), dtype=np.float64)
        delta_state_trajectory[0] = [0.0] * self._n_states

        # This is the updated input trajectory we will return after applying the input perturbations.
        input_next_trajectory = np.nan * np.ones_like(input_trajectory, dtype=np.float64)

        zip_object = zip(
            input_trajectory,
            state_trajectory[:-1],
            state_trajectory[1:],
            lqr_input_policy.state_feedback_matrices,
            lqr_input_policy.feedforward_inputs,
        )

        for input_idx, (input_lin, state_lin, state_lin_next, state_feedback_matrix, feedforward_input) in enumerate(
            zip_object
        ):
            # Compute locally optimal input perturbation.
            delta_state = delta_state_trajectory[input_idx]
            delta_input = state_feedback_matrix @ delta_state + feedforward_input

            # Apply state and input perturbation.
            input_perturbed = input_lin + delta_input
            state_perturbed = state_lin + delta_state
            state_perturbed[2] = principal_value(state_perturbed[2])

            # Run dynamics with perturbed state/inputs to get next state.
            # We get the actually applied input since it might have been clipped/modified to satisfy constraints.
            state_perturbed_next, input_perturbed, _, _ = self._dynamics_and_jacobian(state_perturbed, input_perturbed)

            # Compute next state perturbation given next state.
            delta_state_next = state_perturbed_next - state_lin_next
            delta_state_next[2] = principal_value(delta_state_next[2])

            delta_state_trajectory[input_idx + 1] = delta_state_next
            input_next_trajectory[input_idx] = input_perturbed

        assert ~np.any(np.isnan(input_next_trajectory)), "All next inputs should be valid float values."

        return input_next_trajectory  # type: ignore
