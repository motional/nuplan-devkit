import unittest

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.log_future_planner import LogFuturePlanner
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class TestLogFuturePlanner(unittest.TestCase):
    """
    Test LogFuturePlanner class
    """

    def _get_mock_planner_input(self) -> PlannerInput:
        """
        Returns a mock PlannerInput for testing.
        :return: PlannerInput.
        """
        buffer = SimulationHistoryBuffer.initialize_from_list(
            1, [self.scenario.initial_ego_state], [self.scenario.initial_tracked_objects]
        )
        return PlannerInput(iteration=SimulationIteration(TimePoint(0), 0), history=buffer, traffic_light_data=None)

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.scenario = MockAbstractScenario(number_of_future_iterations=20)
        self.num_poses = 10
        self.future_time_horizon = 5
        self.planner = LogFuturePlanner(self.scenario, self.num_poses, self.future_time_horizon)

    def test_compute_trajectory(self) -> None:
        """Test compute_trajectory"""
        planner_input = self._get_mock_planner_input()
        trajectory = self.planner.compute_trajectory([planner_input])[0]
        self.assertEqual(len(trajectory.get_sampled_trajectory()), self.num_poses + 1)


if __name__ == '__main__':
    unittest.main()
