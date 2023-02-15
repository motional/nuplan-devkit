import unittest

from nuplan.planning.metrics.evaluation_metrics.base.violation_metric_base import ViolationMetricBase
from nuplan.planning.metrics.metric_result import MetricViolation
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario


class TestViolationMetricBase(unittest.TestCase):
    """Creates mock violations for testing."""

    def setUp(self) -> None:
        """Set up mock violations."""
        self.violation_metric_base = ViolationMetricBase(
            name='metric_1',
            category='Dynamics',
            max_violation_threshold=1,
        )
        self.mock_abstract_scenario = MockAbstractScenario()

        self.violation_metric_1 = [
            self._create_mock_violation('metric_1', duration=3, extremum=12.23, mean=8.9),
            self._create_mock_violation('metric_1', duration=1, extremum=123.23, mean=111.1),
            self._create_mock_violation('metric_1', duration=10, extremum=12.23, mean=4.92),
        ]

        self.violation_metric_2 = [self._create_mock_violation('metric_2', duration=13, extremum=1.2, mean=0.0)]

    def _create_mock_violation(self, metric_name: str, duration: int, extremum: float, mean: float) -> MetricViolation:
        """Creates a simple violation
        :param metric_name: name of the metric
        :param duration: duration of the violation
        :param extremum: maximally violating value
        :param mean: mean value of violation depth
        :return: a MetricViolation with the given parameters.
        """
        return MetricViolation(
            metric_computator=self.violation_metric_base.name,
            name=metric_name,
            metric_category=self.violation_metric_base.category,
            unit='unit',
            start_timestamp=0,
            duration=duration,
            extremum=extremum,
            mean=mean,
        )

    def test_successful_aggregation(self) -> None:
        """Checks that the aggregation of MetricViolations works as intended."""
        aggregated_metrics = self.violation_metric_base.aggregate_metric_violations(
            metric_violations=self.violation_metric_1, scenario=self.mock_abstract_scenario
        )[0]
        self.assertEqual(aggregated_metrics.metric_computator, self.violation_metric_base.name)
        self.assertEqual(aggregated_metrics.metric_category, self.violation_metric_base.category)

        statistics = aggregated_metrics.statistics
        self.assertEqual(len(self.violation_metric_1), statistics[0].value)
        self.assertAlmostEqual(statistics[1].value, 123.23, 2)
        self.assertAlmostEqual(statistics[2].value, 12.23, 3)
        self.assertAlmostEqual(statistics[3].value, 13.357, 3)

    def test_failure_on_mixed_metrics(self) -> None:
        """Checks that the aggregation fails when called on MetricViolations from different metrics."""
        with self.assertRaises(AssertionError):
            self.violation_metric_base.aggregate_metric_violations(
                self.violation_metric_1 + self.violation_metric_2, scenario=self.mock_abstract_scenario
            )

    def test_empty_statistics_on_empty_violations(self) -> None:
        """Checks that for an empty list of MetricViolations we get a MetricStatistics with zero violations."""
        empty_statistics = self.violation_metric_base.aggregate_metric_violations([], self.mock_abstract_scenario)[0]
        # Always true for zero violation
        self.assertTrue(empty_statistics.statistics[0].value)


if __name__ == '__main__':
    unittest.main()
