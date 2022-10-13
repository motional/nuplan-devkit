import pathlib
import tempfile
import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

from nuplan.planning.metrics.aggregator.test.mock_abstract_metric_aggregator import MockAbstractMetricAggregator
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback

SCENARIO_NAME = "test_scenario"
PLANNER_NAME = "test_planner"
METRICS_LIST = "MetricsList"


class TestMetricAggregatorCallback(TestCase):
    """Test MetricAggregatorCallback."""

    def setUp(self) -> None:
        """Setup mocks for the tests"""
        self.mock_metric_aggregator_callback = Mock(spec=MetricAggregatorCallback)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp_dir.name)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metric_aggregators = [MockAbstractMetricAggregator(self.path)]

    def tearDown(self) -> None:
        """Clean up tmp dir."""
        self.tmp_dir.cleanup()

    def test_metric_callback_init(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        # Code execution
        metric_aggregator_callback = MetricAggregatorCallback(str(self.path), self.metric_aggregators)

        # Expectations check
        self.assertEqual(metric_aggregator_callback._metric_save_path, self.path)
        self.assertEqual(metric_aggregator_callback._metric_aggregators, self.metric_aggregators)

    @patch('nuplan.planning.simulation.main_callback.metric_aggregator_callback.logger')
    def test_on_run_simulation_end(self, logger: MagicMock) -> None:
        """
        Tests if the callback is called with the correct parameters.
        """
        # Code execution
        metric_file_callback = MetricAggregatorCallback(str(self.path), self.metric_aggregators)
        metric_file_callback.on_run_simulation_end()

        # Expectations check
        logger.warning.assert_has_calls([call('dummy_metric_aggregator: No metric files found for aggregation!')])
        logger.info.assert_has_calls([call('Metric aggregator: 00:00:00 [HH:MM:SS]')])


if __name__ == '__main__':
    unittest.main()
