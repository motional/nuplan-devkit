import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import ILQRSolver
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

DoubleMatrix = npt.NDArray[np.float64]


class ILQRTracker(AbstractTracker):
    """
    Tracker using an iLQR solver with a kinematic bicycle model.
    """

    def __init__(self, n_horizon: int, ilqr_solver: ILQRSolver) -> None:
        """
        Initialize tracker parameters, primarily the iLQR solver.
        :param n_horizon: Maximum time horizon (number of discrete time steps) that we should plan ahead.
                          Please note the associated discretization_time is specified in the ilqr_solver.
        :param ilqr_solver: Solver used to compute inputs to apply.
        """
        assert n_horizon > 0, "The time horizon length should be positive."
        self._n_horizon = n_horizon

        self._ilqr_solver = ilqr_solver

    def track_trajectory(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        initial_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> DynamicCarState:
        """Inherited, see superclass."""
        current_state: DoubleMatrix = np.array(
            [
                initial_state.rear_axle.x,
                initial_state.rear_axle.y,
                initial_state.rear_axle.heading,
                initial_state.dynamic_car_state.rear_axle_velocity_2d.x,
                initial_state.tire_steering_angle,
            ]
        )

        # Determine reference trajectory.  This might be shorter than self._n_horizon states if near trajectory end.
        reference_trajectory = self._get_reference_trajectory(current_iteration, trajectory)

        # Run the iLQR solver to get the optimal input sequence to track the reference trajectory.
        solutions = self._ilqr_solver.solve(current_state, reference_trajectory)
        optimal_inputs = solutions[-1].input_trajectory

        # Extract optimal input to apply at the current timestep.
        accel_cmd = optimal_inputs[0, 0]
        steering_rate_cmd = optimal_inputs[0, 1]

        return DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(accel_cmd, 0),
            tire_steering_rate=steering_rate_cmd,
        )

    def _get_reference_trajectory(
        self, current_iteration: SimulationIteration, trajectory: AbstractTrajectory
    ) -> DoubleMatrix:
        """
        Determines reference trajectory, (z_{ref,k})_k=0^self._n_horizon.
        In case the query timestep exceeds the trajectory length, we return a smaller trajectory (z_{ref,k})_k=0^M,
        where M < self._n_horizon.  The shorter reference will then be handled downstream by the solver appropriately.
        :param current_iteration: Provides the current time from which we interpolate.
        :param trajectory: The full planned trajectory from which we perform state interpolation.
        :return a (M+1 or self._n_horizon+1) by self._n_states array.
        """
        assert trajectory.start_time.time_s <= current_iteration.time_s, "Current time is before trajectory start."
        assert current_iteration.time_s <= trajectory.end_time.time_s, "Current time is after trajectory end"

        discretization_time = self._ilqr_solver._solver_params.discretization_time

        time_deltas_s: DoubleMatrix = np.array(
            [x * discretization_time for x in range(0, self._n_horizon + 1)], dtype=np.float64
        )
        states_interp = []

        for tm_delta_s in time_deltas_s:
            timepoint = TimePoint(int(tm_delta_s * 1e6)) + current_iteration.time_point

            if timepoint > trajectory.end_time:
                break

            state = trajectory.get_state_at_time(timepoint)

            states_interp.append(
                [
                    state.rear_axle.x,
                    state.rear_axle.y,
                    state.rear_axle.heading,
                    state.dynamic_car_state.rear_axle_velocity_2d.x,
                    state.tire_steering_angle,
                ]
            )

        return np.array(states_interp)
