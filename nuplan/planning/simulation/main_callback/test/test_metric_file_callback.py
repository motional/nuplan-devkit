import pathlib
import tempfile
import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback

SCENARIO_NAME = "test_scenario"
PLANNER_NAME = "test_planner"
METRICS_LIST = "MetricsList"


class TestMetricFileCallback(TestCase):
    """Tests metrics files generation at the end fo the simulation."""

    def setUp(self) -> None:
        """Setup mocks for the tests"""
        self.mock_metric_file_callback = Mock(spec=MetricFileCallback)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp_dir.name)
        self.path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up tmp dir."""
        self.tmp_dir.cleanup()

    def test_metric_callback_init(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        # Code execution
        metric_file_callback = MetricFileCallback(
            metric_file_output_path=self.tmp_dir.name, scenario_metric_paths=[self.tmp_dir.name]
        )

        # Expectations check
        self.assertEqual(metric_file_callback._metric_file_output_path, self.path)
        self.assertEqual(metric_file_callback._scenario_metric_paths, [self.path])

    @patch('nuplan.planning.simulation.main_callback.metric_file_callback.logger')
    def test_on_run_simulation_end(self, logger: MagicMock) -> None:
        """
        Tests if the callback is called with the correct parameters.
        """
        # Code execution
        metric_file_callback = MetricFileCallback(
            metric_file_output_path=self.tmp_dir.name, scenario_metric_paths=[self.tmp_dir.name]
        )
        metric_file_callback.on_run_simulation_end()

        # Expectations check
        logger.info.assert_has_calls([call('Metric files integration: 00:00:00 [HH:MM:SS]')])


if __name__ == '__main__':
    unittest.main()
