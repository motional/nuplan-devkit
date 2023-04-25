import time
import unittest
from unittest.mock import Mock, patch

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.log_future_planner import LogFuturePlanner
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
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

    def test_name(self) -> None:
        """Tests planner name is set correctly."""
        # Call method under test
        result = self.planner.name()

        # Assertions
        self.assertEqual(result, "LogFuturePlanner")

    @patch('nuplan.planning.simulation.planner.log_future_planner.DetectionsTracks')
    def test_observation_type(self, mock_detection_tracks: Mock) -> None:
        """Tests observation type is set correctly."""
        # Call method under test
        result = self.planner.observation_type()

        # Assertions
        self.assertEqual(result, mock_detection_tracks)

    def test_compute_trajectory(self) -> None:
        """Test compute_trajectory"""
        planner_input = self._get_mock_planner_input()
        start_time = time.perf_counter()

        # Call method under test
        result = self.planner.compute_trajectory(planner_input)

        compute_trajectory_time = time.perf_counter() - start_time
        self.assertEqual(len(result.get_sampled_trajectory()), self.num_poses + 1)

        # Basic sanity checks on the planner report
        planner_report = self.planner.generate_planner_report()
        self.assertEqual(len(planner_report.compute_trajectory_runtimes), 1)
        self.assertNotIsInstance(planner_report, MLPlannerReport)
        self.assertAlmostEqual(planner_report.compute_trajectory_runtimes[0], compute_trajectory_time, delta=0.1)

    def test_compute_trajectory_fail_extraction_previous_available(self) -> None:
        """
        Test compute_trajectory when future ego extraction from scenario fails and planner should fall back on previous
        trajectory.
        """
        previous_trajectory = Mock()
        self.planner._trajectory = previous_trajectory
        planner_input = self._get_mock_planner_input()

        with patch.object(self.scenario, 'get_ego_future_trajectory', side_effect=AssertionError):
            # Call method under test
            result = self.planner.compute_trajectory(planner_input)

        # Should fall back on previous trajectory
        self.assertEqual(result, previous_trajectory)

    def test_compute_trajectory_fail_extraction_no_previous(self) -> None:
        """
        Test compute_trajectory when future ego extraction from scenario fails and there is no prior trajectory
        to fall back on.
        """
        self.planner._trajectory = None
        planner_input = self._get_mock_planner_input()

        with patch.object(self.scenario, 'get_ego_future_trajectory', side_effect=AssertionError):
            with self.assertRaises(RuntimeError):
                # Call method under test
                _ = self.planner.compute_trajectory(planner_input)


if __name__ == '__main__':
    unittest.main()
