import logging
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.interpolate as sp_interp
import sympy as sym
from control import StateSpace, ctrb, dlqr
from sympy import Matrix, cos, sin, tan

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.database.utils.measure import angle_diff
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

logger = logging.getLogger(__name__)


class StateIndex(IntEnum):
    """
    Index mapping for the state vector
    """

    X_POS = 0  # [m] The x position in global coordinates.
    Y_POS = 1  # [m] The y position in global coordinates.
    HEADING = 2  # [rad] The yaw in global coordinates.
    VELOCITY = 3  # [m/s] The velocity along the longitudinal axis of the vehicles.
    STEERING_ANGLE = 4  # [rad] The wheel angle relative to the longitudinal axis of the vehicle.


class InputIndex(IntEnum):
    """
    Index mapping for the input vector
    """

    ACCEL = 0  # [m/s^2] The acceleration along the longitudinal axis of the vehicles.
    STEERING_RATE = 1  # [rad/s] The steering rate.


class LQRTracker(AbstractTracker):
    """
    Implements an LQR tracker for a kinematic bicycle model.
    States: [x, y, heading, velocity, steering_angle]
    Inputs: [acceleration, steering_rate]
    Dynamics:
        x_dot              = velocity * cos(heading)
        y_dot              = velocity * sin(heading)
        heading_dot        = velocity * tan(steering_angle) / car_length
        velocity_dot       = acceleration
        steering_angle_dot = steering_rate

    The state space is discretized using zoh and the discrete ARE is solved.
    The control action passed on to the motion model are:
        - acceleration
        - steering angle

    Integral Action Augmentation:
    The system dynamics are augmented introduce integral action.

    x_dot = Ax + Bu
    z_dot = Cx - r   # integral of output (error)

    Augmented discrete state space:
    x_hat_dot = | A 0||x| + |B|u
                |-C I||z| + |0|
    """

    def __init__(
        self,
        q_diag: npt.NDArray[np.float64],
        r_diag: npt.NDArray[np.float64],
        proportional_gain: float,
        look_ahead_seconds: float,
        look_ahead_meters: float,
        stopping_velocity: float,
        vehicle: VehicleParameters = get_pacifica_parameters(),
    ):
        """
        Constructor for LQR controller
        :param q_diag: The diagonal terms of the Q matrix
        :param r_diag: The diagonal terms of the R matrix
        :param proportional_gain: The proportional_gain term for the P controller
        :param look_ahead_seconds: [s] The lookahead time
        :param look_ahead_meters: [m] The lookahead distance
        :param vehicle: Vehicle parameters
        """
        assert len(q_diag) == 6, "q_diag has to have length of 6"
        assert len(r_diag) == 2, "r_diag has to have length of 2"
        assert proportional_gain >= 0, "proportional_gain has to be greater than 0"
        assert look_ahead_seconds >= 0, "look_ahead_seconds has to be greater than 0"
        assert look_ahead_meters >= 0, "look_ahead_meters has to be greater than 0"
        assert stopping_velocity >= 0, "stopping_velocity has to be greater than 0"

        self._epsilon = 1e-6
        self._vehicle = vehicle

        # Controller parameters
        self._q_matrix: npt.NDArray[np.float64] = np.diag(q_diag)
        self._r_matrix: npt.NDArray[np.float64] = np.diag(r_diag)
        self._proportional_gain = proportional_gain
        self._look_ahead_seconds = look_ahead_seconds
        self._look_ahead_meters = look_ahead_meters
        self._stopping_velocity = stopping_velocity

        # set to None to allow lazily loading of jacobians
        self._a_matrix: Optional[Matrix] = None
        self._b_matrix: Optional[Matrix] = None

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._a_matrix, self._b_matrix = self._compute_linear_state_space()

    def track_trajectory(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        initial_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> DynamicCarState:
        """Inherited, see superclass."""
        sampling_time = next_iteration.time_point.time_s - current_iteration.time_point.time_s
        state_vector: npt.NDArray[np.float64] = np.array(
            [
                initial_state.rear_axle.x,
                initial_state.rear_axle.y,
                initial_state.rear_axle.heading,
                initial_state.dynamic_car_state.rear_axle_velocity_2d.x,
                initial_state.tire_steering_angle,
            ]
        )
        # Dynamically reduce the lookahead with velocity
        # At high velocity the reference pose at the next time step becomes further away from the state.
        # This in turn induces a large error and hence the LQR commands large acceleration.
        # Hence, we must reduce the lookahead to keep the distance from the LQR state bounded.
        look_ahead_seconds = min(
            self._look_ahead_meters / (abs(state_vector[StateIndex.VELOCITY]) + self._epsilon), self._look_ahead_seconds
        )

        try:
            sample_time = current_iteration.time_point + TimePoint(int(look_ahead_seconds * 1e6))
            next_state = trajectory.get_state_at_time(sample_time)
            reference_velocity = self._infer_refernce_velocity(trajectory, sample_time)

        except AssertionError as e:
            raise AssertionError("Lookahead time exceeds trajectory length!") from e

        reference_vector: npt.NDArray[np.float64] = np.array(
            [
                next_state.rear_axle.x,
                next_state.rear_axle.y,
                next_state.rear_axle.heading,
                reference_velocity,
                next_state.tire_steering_angle,
            ]
        )

        accel_cmd, steering_rate_cmd = self._compute_control_action(
            state_vector, reference_vector, initial_state, sampling_time
        )

        return DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(accel_cmd, 0),
            tire_steering_rate=steering_rate_cmd,
        )

    @staticmethod
    def _infer_refernce_velocity(trajectory: AbstractTrajectory, sample_time: TimePoint) -> float:
        """
        Calculates the reference velocity from the give pose trajectory.
        :param trajectory: The reference trajectory to track.
        :param sample_time: The time point to sample the trajectory.
        :return: [m/s] The velocity reference.
        """
        sampled_ego_trajectory = trajectory.get_sampled_trajectory()
        rear_axle_poses: npt.NDArray[np.int32] = np.array(
            [[*sample.rear_axle.point] for sample in sampled_ego_trajectory]
        )
        time_point: npt.NDArray[np.int32] = np.array([sample.time_point.time_us for sample in sampled_ego_trajectory])
        approx_vel = np.diff(rear_axle_poses.transpose()) / np.diff(time_point * 1e-6)
        interp = sp_interp.interp1d(time_point[:-1], approx_vel, axis=1)
        return float(np.hypot(*interp(sample_time.time_us)))

    def _compute_control_action(
        self,
        state: npt.NDArray[np.float64],
        reference: npt.NDArray[np.float64],
        ego_state: EgoState,
        sampling_time: float,
    ) -> Tuple[float, float]:
        """
        Computes the control action given a state and a reference.
        :param state: The current state.
        :param reference: The reference to track.
        :param ego_state: The ego state.
        :param sampling_time: [s] The sampling interval.
        :return: The control actions ([m/s^2] accel, [rad/s] steering rate)
        """
        tracking_error: npt.NDArray[np.float64] = self._compute_error(state, reference, sampling_time)

        # Stopping case
        if (
            reference[StateIndex.VELOCITY] < self._stopping_velocity
            and state[StateIndex.VELOCITY] < self._stopping_velocity
        ):
            # Apply proportional controller
            accel = -self._proportional_gain * tracking_error[StateIndex.VELOCITY]
            return accel, 0.0

        return self._compute_lqr_control_action(tracking_error, ego_state, sampling_time)

    def _compute_lqr_control_action(
        self, tracking_error: npt.NDArray[np.float64], ego_state: EgoState, sampling_time: float
    ) -> Tuple[float, float]:
        """
        Compute the control action using LQR policy
        :param tracking_error: <np.ndarray: num_states> The tracking error vector.
        :param ego_state: The ego state.
        :param sampling_time: [s] The sampling interval.
        :return: The control actions ([m/s^2] accel, [rad/s] steering rate)
        """
        # Linearize system around the current ego state
        a_matrix, b_matrix = self._linearize_model(
            heading=ego_state.rear_axle.heading,
            velocity=ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            steering_angle=ego_state.tire_steering_angle,
        )

        # Discretize System
        sys = StateSpace(a_matrix, b_matrix, np.zeros_like(a_matrix), np.zeros_like(b_matrix))
        discrete_sys = sys.sample(sampling_time)

        try:
            # Solve the discrete lqr problem
            gain_matrix, _, _ = dlqr(discrete_sys, self._q_matrix, self._r_matrix)
            # Control feedback law u[t+1] = -Ku[t]
            control_action = -gain_matrix.dot(tracking_error)

        except np.linalg.LinAlgError:
            logger.warning(
                "Failed to solve for LQR gain matrix. Applying zero action \n "
                "Controllability matrix not full rank: %d < 5\n"
                "Linearization point [heading, velocity, steering angle: [%0.2f, %0.2f, %0.2f]",
                np.linalg.matrix_rank(ctrb(a_matrix, b_matrix)),
                ego_state.rear_axle.heading,
                ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                ego_state.tire_steering_angle,
            )
            control_action = np.array([0.0, 0.0])

        return control_action[InputIndex.ACCEL], control_action[InputIndex.STEERING_RATE]

    @staticmethod
    def _compute_error(
        state: npt.NDArray[np.float64], reference: npt.NDArray[np.float64], sampling_time: float
    ) -> npt.NDArray[np.float64]:
        """
        Compute the error between the state and reference.
        :param state: State vector.
        :param reference: Reference vector.
        :param sampling_time: [s] The sampling interval.
        """
        error: npt.NDArray[np.float64] = state - reference

        # Handle angular states
        error[StateIndex.HEADING] = angle_diff(state[StateIndex.HEADING], reference[StateIndex.HEADING], 2 * np.pi)
        error[StateIndex.STEERING_ANGLE] = angle_diff(
            state[StateIndex.STEERING_ANGLE], reference[StateIndex.STEERING_ANGLE], 2 * np.pi
        )

        # Handle error integral term
        integral_term = error * sampling_time
        return np.concatenate((error, [integral_term[StateIndex.VELOCITY]]))

    def _linearize_model(
        self, heading: float, velocity: float, steering_angle: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Linearizes the kinematic bicycle model around the given operating point.
        :param heading: [rad] heading linearization point.
        :param velocity: [m/s] velocity linearization point.
        :param steering_angle: [rad] steering_angle linearization point.
        :return: The A and B matrices of the linearized dynamics of the kinematic bicycle model x_dot = Ax(t) + Bu(t).
        """
        if self._a_matrix is None or self._b_matrix is None:
            raise ValueError("A and B state space matrices are None. Need to initialize the tracker first!")

        # Substitute in linearization point
        a_matrix = self._a_matrix.subs([('theta', heading), ('vel', velocity), ('delta', steering_angle)])

        a_matrix = np.array(a_matrix).astype(np.float64)
        b_matrix: npt.NDArray[np.float64] = np.array(self._b_matrix).astype(np.float64)

        # Augment the system with integrators
        num_states = a_matrix.shape[0]
        num_inputs = b_matrix.shape[1]

        # Process the states to be integrated
        c_matrix = np.zeros((1, len(StateIndex)))
        c_matrix[0][StateIndex.VELOCITY] = 1.0
        num_integrators = c_matrix.shape[0]

        a_matrix = np.block([[a_matrix, np.zeros((num_states, num_integrators))], [-c_matrix, np.eye(num_integrators)]])
        b_matrix = np.vstack([b_matrix, np.zeros((num_integrators, num_inputs))])

        return a_matrix, b_matrix

    def _compute_linear_state_space(self) -> Tuple[Matrix, Matrix]:
        """
        Calculate the A and B jacobian of a kinematic bicycle model where the reference is at the rear axle.
        Where A is the jacobian wrt to the state vector and B jacobian is wrt to the input vector. Together they
        describe the linear system model x_dot = Ax(t) + Bu(t).
        :returns: A and B jacobians.
        """
        # State vector
        x = sym.Symbol('x')  # [m] x position in world frame
        y = sym.Symbol('y')  # [m] y position in world frame
        theta = sym.Symbol('theta')  # [rad] Heading at the rear axle
        vel = sym.Symbol('vel')  # [m/s] velocity
        delta = sym.Symbol('delta')  # [rad] steering angle

        state_vector = Matrix([x, y, theta, vel, delta])

        # Input vector
        accel = sym.Symbol('accel')  # [m/s^2] acceleration
        phi = sym.Symbol('phi')  # [rad/s] steering rate
        input_vector = Matrix([accel, phi])

        # Dynamics
        x_dot = vel * cos(theta)
        y_dot = vel * sin(theta)
        theta_dot = vel * tan(delta) / self._vehicle.wheel_base

        ode = Matrix([x_dot, y_dot, theta_dot, accel, phi])

        return ode.jacobian(state_vector), ode.jacobian(input_vector)
