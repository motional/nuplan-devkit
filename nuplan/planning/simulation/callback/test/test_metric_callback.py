import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.simulation.callback.metric_callback import MetricCallBack
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

SCENARIO_NAME = "test_scenario"
PLANNER_NAME = "test_planner"
METRICS_LIST = "MetricsList"


class TestMetricCallBack(TestCase):
    def setUp(self) -> None:
        """
        Setup mocks for the tests
        """
        self.mock_metric_engine = Mock(spec=MetricsEngine)
        self.mock_metric_engine.compute = Mock(return_value=METRICS_LIST)
        self.mock_setup = Mock()
        self.mock_planner = Mock(spec=AbstractPlanner)
        self.mock_planner.name = Mock(return_value=PLANNER_NAME)
        self.mock_history = Mock()

        return super().setUp()

    def test_metric_callback_init(self) -> None:
        """
        Tests if all the properties are set to the expected values in constructor.
        """
        # Code execution
        mc = MetricCallBack(
            self.mock_metric_engine, SCENARIO_NAME)

        # Expectations check
        self.assertEqual(mc._metric_engine, self.mock_metric_engine)
        self.assertEqual(mc._scenario_name, SCENARIO_NAME)

    @patch('nuplan.planning.simulation.callback.metric_callback.logger')
    def test_on_simulation_end(self, logger: MagicMock) -> None:
        """
        Tests if the metric engine compute is called with the correct parameters.
        Tests if the metric engine save_metric_files is called with compute's result.
        Tests if the logger is called with the correct parameters.
        """
        # Code execution
        mc = MetricCallBack(self.mock_metric_engine, SCENARIO_NAME)
        mc.on_simulation_end(
            self.mock_setup, self.mock_planner, self.mock_history)

        # Expectations check
        logger.info.assert_has_calls([
            call("Starting metrics computation..."),
            call("Finished metrics computation!"),
            call("Saved metrics!")
        ])

        self.mock_planner.name.assert_called_once()
        self.mock_metric_engine.compute.assert_called_once_with(
            self.mock_history, scenario_name=SCENARIO_NAME, planner_name=PLANNER_NAME)
        self.mock_metric_engine.save_metric_files.assert_called_once_with(
            METRICS_LIST)


if __name__ == '__main__':
    unittest.main()
