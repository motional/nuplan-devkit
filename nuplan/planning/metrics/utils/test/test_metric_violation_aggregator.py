import unittest

from nuplan.planning.metrics.metric_result import MetricStatisticsType, MetricViolation
from nuplan.planning.metrics.utils.metric_violation_aggregator import aggregate_metric_violations


def create_mock_violation(metric_name: str, duration: int, extremum: float, mean: float) -> MetricViolation:
    """ Creates a simple violation.

    :param metric_name: name of the metric
    :param duration: duration of the violation
    :param extremum: maximally violating value
    :param mean: mean value of violation depth
    :return: a MetricViolation with the given parameters.
    """
    return MetricViolation("metric_computator", metric_name, "metric_category", "unit", 0, duration, extremum, mean)


class TestMetricViolationAggregator(unittest.TestCase):

    def setUp(self) -> None:
        """ Creates mock violations for testing. """
        self.violation_metric_1 = [create_mock_violation("metric_1", duration=3, extremum=12.23, mean=8.9),
                                   create_mock_violation("metric_1", duration=1, extremum=123.23, mean=111.1),
                                   create_mock_violation("metric_1", duration=10, extremum=12.23, mean=4.92),
                                   ]

        self.violation_metric_2 = [create_mock_violation("metric_2", duration=13, extremum=1.2, mean=0.0)]

    def test_successful_aggregation(self) -> None:
        """ Checks that the aggregation of MetricViolations works as intended. """
        aggregated_metrics = aggregate_metric_violations(self.violation_metric_1,
                                                         self.violation_metric_1[0].metric_computator,
                                                         self.violation_metric_1[0].metric_category,
                                                         self.violation_metric_1[0].name)
        self.assertEqual(aggregated_metrics.metric_computator, self.violation_metric_1[0].metric_computator)
        self.assertEqual(aggregated_metrics.metric_category, self.violation_metric_1[0].metric_category)
        self.assertEqual(aggregated_metrics.name, self.violation_metric_1[0].name)

        statistics = aggregated_metrics.statistics
        self.assertAlmostEqual(123.23, statistics[MetricStatisticsType.MAX].value, 2)
        self.assertEqual(len(self.violation_metric_1), statistics[MetricStatisticsType.COUNT].value)
        self.assertAlmostEqual(13.357, statistics[MetricStatisticsType.MEAN].value, 3)

    def test_failure_on_mixed_metrics(self) -> None:
        """ Checks that the aggregation fails when called on MetricViolations from different metrics. """
        with self.assertRaises(AssertionError):
            aggregate_metric_violations(self.violation_metric_1 + self.violation_metric_2, "", "", "")

    def test_empty_statistics_on_empty_violations(self) -> None:
        """ Checks that for an empty list of MetricViolations we get a MetricStatistics with zero violations. """
        empty_statistics = aggregate_metric_violations([], "foo", "bar", "baz")
        assert empty_statistics.metric_computator == "foo"
        assert empty_statistics.metric_category == "bar"
        assert empty_statistics.name == "baz"
        assert empty_statistics.statistics[MetricStatisticsType.COUNT].value == 0


if __name__ == '__main__':
    unittest.main()
