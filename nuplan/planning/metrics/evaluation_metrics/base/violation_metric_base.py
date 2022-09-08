from typing import List, Optional

import numpy as np

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import (
    MetricStatistics,
    MetricStatisticsType,
    MetricViolation,
    Statistic,
    TimeSeries,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class ViolationMetricBase(MetricBase):
    """Base class for evaluation of violation metrics."""

    def __init__(
        self, name: str, category: str, max_violation_threshold: int = 0, metric_score_unit: Optional[str] = None
    ) -> None:
        """
        Initializes the ViolationMetricBase class
        :param name: Metric name
        :param category: Metric category
        :param max_violation_threshold: Maximum threshold for the violation when computing the score.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._max_violation_threshold = max_violation_threshold
        self.number_of_violations = 0

    def aggregate_metric_violations(
        self,
        metric_violations: List[MetricViolation],
        scenario: AbstractScenario,
        time_series: Optional[TimeSeries] = None,
    ) -> List[MetricStatistics]:
        """
        Aggregates (possibly) multiple MetricViolations to a MetricStatistics.
        All the violations must be of the same metric.
        :param metric_violations: The list of violations for a single metric name.
        :param scenario: Scenario running this metric.
        :param time_series: Time series metrics.
        :return Statistics about the violations.
        """
        if not metric_violations:
            statistics = [
                Statistic(
                    name=f'{self.name}',
                    unit=MetricStatisticsType.BOOLEAN.unit,
                    value=True,
                    type=MetricStatisticsType.BOOLEAN,
                ),
            ]

        else:
            sample_violation = metric_violations[0]
            name = sample_violation.name
            unit = sample_violation.unit

            extrema = []
            mean_values = []
            durations = []
            for violation in metric_violations:
                # We don't want to aggregate different metrics
                assert name == violation.name
                extrema.append(violation.extremum)
                mean_values.append(violation.mean)
                durations.append(violation.duration)

            max_val = max(extrema)
            min_val = min(extrema)

            # Violations which are instantaneous will be reported with unitary duration, while violations which span
            # over time have the respective duration.
            # If a violation is detected at the last time step will have duration 0
            # and won't be taken into account for the mean.
            mean_val = np.sum([mean_value * duration for mean_value, duration in zip(mean_values, durations)]) / sum(
                durations
            )

            statistics = [
                Statistic(
                    name=f'number_of_violations_of_{self.name}',
                    unit=MetricStatisticsType.COUNT.unit,
                    value=len(metric_violations),
                    type=MetricStatisticsType.COUNT,
                ),
                Statistic(
                    name=f'max_violation_of_{self.name}', unit=unit, value=max_val, type=MetricStatisticsType.MAX
                ),
                Statistic(
                    name=f'min_violation_of_{self.name}', unit=unit, value=min_val, type=MetricStatisticsType.MIN
                ),
                Statistic(
                    name=f'mean_violation_of_{self.name}', unit=unit, value=mean_val, type=MetricStatisticsType.MEAN
                ),
                Statistic(
                    name=f'{self.name}',
                    unit=MetricStatisticsType.BOOLEAN.unit,
                    value=False,
                    type=MetricStatisticsType.BOOLEAN,
                ),
            ]

        self.number_of_violations = len(metric_violations)
        results: list[MetricStatistics] = self._construct_metric_results(
            metric_statistics=statistics,
            scenario=scenario,
            time_series=time_series,
            metric_score_unit=self.metric_score_unit,
        )
        return results

    def _compute_violation_metric_score(self, number_of_violations: int) -> float:
        """
        Compute a metric score based on a violation threshold. It is 1 - (x / (max_violation_threshold + 1))
        The score will be 0 if the number of violations exceeds this value
        :param number_of_violations: Total number of violations
        :return A metric score between 0 and 1.
        """
        return max(0.0, 1.0 - (number_of_violations / (self._max_violation_threshold + 1)))

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return self._compute_violation_metric_score(number_of_violations=self.number_of_violations)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        # Subclasses should implement this
        raise NotImplementedError
