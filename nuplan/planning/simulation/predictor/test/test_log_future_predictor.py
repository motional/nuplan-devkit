import time
import unittest

from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.predictor.log_future_predictor import LogFuturePredictor
from nuplan.planning.simulation.predictor.predictor_report import MLPredictorReport
from nuplan.planning.simulation.predictor.test.mock_abstract_predictor import get_mock_predictor_input
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class TestLogFuturePredictor(unittest.TestCase):
    """
    Test LogFuturePredictor class.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.scenario = MockAbstractScenario()
        self.future_trajectory_sampling = TrajectorySampling(num_poses=1, time_horizon=1.0)
        self.predictor = LogFuturePredictor(self.scenario, self.future_trajectory_sampling)

    def test_compute_predicted_trajectories(self) -> None:
        """Test compute_predicted_trajectories."""
        predictor_input = get_mock_predictor_input()
        start_time = time.perf_counter()
        detections = self.predictor.compute_predictions(predictor_input)
        compute_predictions_time = time.perf_counter() - start_time

        # Check number of tracked objects stays the same
        _, input_detections = predictor_input.history.current_state
        self.assertEqual(len(detections.tracked_objects), len(input_detections.tracked_objects))

        # Check valid predictions
        for agent in detections.tracked_objects.get_agents():
            self.assertTrue(agent.predictions is not None)
            for prediction in agent.predictions:
                self.assertEqual(len(prediction.valid_waypoints), self.future_trajectory_sampling.num_poses)

        # Basic sanity checks on the predictor report
        predictor_report = self.predictor.generate_predictor_report()
        self.assertEqual(len(predictor_report.compute_predictions_runtimes), 1)
        self.assertNotIsInstance(predictor_report, MLPredictorReport)
        self.assertAlmostEqual(predictor_report.compute_predictions_runtimes[0], compute_predictions_time, delta=0.1)


if __name__ == '__main__':
    unittest.main()
