import unittest

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class TestPerfectTracking(unittest.TestCase):
    """
    Tests Tracker
    """

    def test_perfect_tracker(self) -> None:
        """
        Test the basic functionality of perfect tracker
        """
        initial_time_point = TimePoint(0)
        scenario = MockAbstractScenario(initial_time_us=initial_time_point)
        trajectory = InterpolatedTrajectory(list(scenario.get_expert_ego_trajectory()))
        tracker = PerfectTrackingController(scenario)

        # Check initial state
        desired_state = scenario.initial_ego_state
        state = scenario.initial_ego_state
        self.assertAlmostEqual(state.rear_axle.x, desired_state.rear_axle.x)
        self.assertAlmostEqual(state.rear_axle.y, desired_state.rear_axle.y)
        self.assertAlmostEqual(state.rear_axle.heading, desired_state.rear_axle.heading)

        # Check two steps ahead
        tracker.update_state(
            current_iteration=SimulationIteration(time_point=initial_time_point, index=0),
            next_iteration=SimulationIteration(time_point=TimePoint(int(1 * 1e6)), index=1),
            ego_state=scenario.initial_ego_state,
            trajectory=trajectory,
        )
        next_state = tracker.get_state()
        desired_state = scenario.get_ego_state_at_iteration(2)
        self.assertAlmostEqual(next_state.rear_axle.x, desired_state.rear_axle.x)
        self.assertAlmostEqual(next_state.rear_axle.y, desired_state.rear_axle.y)
        self.assertAlmostEqual(next_state.rear_axle.heading, desired_state.rear_axle.heading)


if __name__ == '__main__':
    unittest.main()
