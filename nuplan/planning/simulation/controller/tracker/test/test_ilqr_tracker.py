import unittest

import numpy as np
import numpy.testing as np_test

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    ILQRSolver,
    ILQRSolverParameters,
    ILQRWarmStartParameters,
)
from nuplan.planning.simulation.controller.tracker.ilqr_tracker import ILQRTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class TestILQRTracker(unittest.TestCase):
    """
    Tests the functionality of the ILQRTracker class.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        # Set up the planned trajectory for the tracker.  We set the initial time point to a nonzero value,
        # as we want to test edge cases where the simulation time is before the trajectory start time.
        self.initial_time_point = TimePoint(1000000)
        self.scenario = MockAbstractScenario(initial_time_us=self.initial_time_point)
        self.trajectory = InterpolatedTrajectory(list(self.scenario.get_expert_ego_trajectory()))

        # Set up the tracker.
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

        self.tracker = ILQRTracker(
            n_horizon=40,
            ilqr_solver=ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params),
        )

        self.discretization_time_us = int(1e6 * self.tracker._ilqr_solver._solver_params.discretization_time)

    def test_track_trajectory(self) -> None:
        """Ensure that we can run a single solver call to track a trajectory."""
        current_iteration = SimulationIteration(time_point=self.initial_time_point, index=0)
        time_point_delta = TimePoint(self.discretization_time_us)
        next_iteration = SimulationIteration(time_point=self.initial_time_point + time_point_delta, index=1)

        self.tracker.track_trajectory(
            current_iteration=current_iteration,
            next_iteration=next_iteration,
            initial_state=self.scenario.initial_ego_state,
            trajectory=self.trajectory,
        )

    def test__get_reference_trajectory(self) -> None:
        """Test reference trajectory extraction for the solver."""
        # (1) Check the cases where the current time is outside the defined trajectory.
        current_iteration_before_trajectory_start = SimulationIteration(
            time_point=self.trajectory.start_time - TimePoint(1), index=0
        )
        with self.assertRaises(AssertionError):
            self.tracker._get_reference_trajectory(current_iteration_before_trajectory_start, self.trajectory)

        current_iteration_after_trajectory_end = SimulationIteration(
            time_point=self.trajectory.end_time + TimePoint(1), index=0
        )
        with self.assertRaises(AssertionError):
            self.tracker._get_reference_trajectory(current_iteration_after_trajectory_end, self.trajectory)

        # (2) Nominal case where the current time is within the trajectory time interval.
        start_time_us = self.trajectory.start_time.time_us
        end_time_us = self.trajectory.end_time.time_us
        mid_time_us = int((start_time_us + end_time_us) / 2)

        for test_time_us in [start_time_us, mid_time_us, end_time_us]:
            expected_trajectory_length = min(
                (end_time_us - test_time_us) // self.discretization_time_us + 1,
                self.tracker._n_horizon + 1,
            )

            current_iteration = SimulationIteration(time_point=TimePoint(test_time_us), index=0)
            reference_trajectory = self.tracker._get_reference_trajectory(current_iteration, self.trajectory)

            self.assertEqual(len(reference_trajectory), expected_trajectory_length)

            first_state_reference_trajectory = reference_trajectory[0]
            first_ego_state_expected = self.trajectory.get_state_at_time(current_iteration.time_point)

            np_test.assert_allclose(first_state_reference_trajectory[0], first_ego_state_expected.rear_axle.x)
            np_test.assert_allclose(first_state_reference_trajectory[1], first_ego_state_expected.rear_axle.y)
            np_test.assert_allclose(first_state_reference_trajectory[2], first_ego_state_expected.rear_axle.heading)
            np_test.assert_allclose(
                first_state_reference_trajectory[3], first_ego_state_expected.dynamic_car_state.rear_axle_velocity_2d.x
            )
            np_test.assert_allclose(first_state_reference_trajectory[4], first_ego_state_expected.tire_steering_angle)


if __name__ == "__main__":
    unittest.main()
