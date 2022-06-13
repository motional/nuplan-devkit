from typing import Dict, List, Optional

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

    def __init__(self, name: str, category: str, max_violation_threshold: int) -> None:
        """
        Initializes the ViolationMetricBase class
        :param name: Metric name
        :param category: Metric category
        :param max_violation_threshold: Maximum threshold for the violation when computing the score.
        """
        super().__init__(name=name, category=category)
        self._max_violation_threshold = max_violation_threshold

    def aggregate_metric_violations(
        self, metric_violations: List[MetricViolation], scenario: AbstractScenario, time_series: TimeSeries = None
    ) -> List[MetricStatistics]:
        """
        Aggregates (possibly) multiple MetricViolations to a MetricStatistics
        All the violations must be of the same metric
        :param metric_violations: The list of violations for a single metric name
        :param scenario: Scenario running this metric
        :return Statistics about the violations.
        """
        if not metric_violations:
            statistics = {
                MetricStatisticsType.COUNT: Statistic(name=f'number_of_{self.name}', unit='count', value=0),
                MetricStatisticsType.BOOLEAN: Statistic(
                    name=f'no_{self.name}',
                    unit='boolean',
                    value=True,
                ),
            }

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

            statistics = {
                MetricStatisticsType.MAX: Statistic(name=f'max_violating_{self.name}', unit=unit, value=max_val),
                MetricStatisticsType.MIN: Statistic(name=f'min_violating_{self.name}', unit=unit, value=min_val),
                MetricStatisticsType.MEAN: Statistic(name=f'mean_{self.name}', unit=unit, value=mean_val),
                MetricStatisticsType.COUNT: Statistic(
                    name=f'number_of_{self.name}',
                    unit='count',
                    value=len(metric_violations),
                ),
                MetricStatisticsType.BOOLEAN: Statistic(
                    name=f'no_{self.name}',
                    unit='boolean',
                    value=False,
                ),
            }

        results = self._construct_metric_results(
            metric_statistics=statistics, scenario=scenario, time_series=time_series
        )

        return results  # type: ignore

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
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return self._compute_violation_metric_score(
            number_of_violations=metric_statistics[MetricStatisticsType.COUNT].value
        )

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        # Subclasses should implement this
        raise NotImplementedError
