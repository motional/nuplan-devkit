from typing import Dict, List, Optional

import numpy as np

from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class MetricBase(AbstractMetricBuilder):
    """Base class for evaluation of metrics."""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializer for MetricBase
        :param name: Metric name
        :param category: Metric category.
        """
        self._name = name
        self._category = category

    @property
    def name(self) -> str:
        """
        Returns the metric name
        :return the metric name.
        """
        return self._name

    @property
    def category(self) -> str:
        """
        Returns the metric category
        :return the metric category.
        """
        return self._category

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> Optional[float]:
        """Inherited, see superclass."""
        # Default score, set None not to use this metric score in aggregating
        return None

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        # Subclasses should implement this
        raise NotImplementedError

    def _compute_time_series_statistic(
        self, time_series: TimeSeries, statistics_type_list: Optional[List[str]] = None
    ) -> Dict[str, Statistic]:
        """
        Compute metric statistics in time series
        :param time_series: time series (with float values)
        :param statistics_type_list: List of available types such as [MetricStatisticsType.MAX,
         MetricStatisticsType.MIN, MetricStatisticsType.MEAN, MetricStatisticsType.P90]. Use all if set to None
        :return A dictionary of metric statistics
        """
        values = time_series.values
        unit = time_series.unit

        if statistics_type_list is None:
            statistics_type_list = [
                MetricStatisticsType.MAX,
                MetricStatisticsType.MIN,
                MetricStatisticsType.MEAN,
                MetricStatisticsType.P90,
            ]
        statistics = {}
        for statistics_type in statistics_type_list:
            if statistics_type == MetricStatisticsType.MAX:
                name = f"max_{self.name}"
                value = np.amax(values)
            elif statistics_type == MetricStatisticsType.MEAN:
                name = f"avg_{self.name}"
                value = np.mean(values)
            elif statistics_type == MetricStatisticsType.MIN:
                name = f"min_{self.name}"
                value = np.min(values)
            elif statistics_type == MetricStatisticsType.P90:
                name = f"p90_{self.name}"
                value = np.percentile(values, 90)
            else:
                raise TypeError('Other metric types statistics cannot be created by compute_statistics()')

            statistics[statistics_type] = Statistic(name=name, unit=unit, value=value)

        return statistics

    def _construct_metric_results(
        self,
        metric_statistics: Dict[str, Statistic],
        scenario: AbstractScenario,
        time_series: Optional[TimeSeries] = None,
    ) -> List[MetricStatistics]:
        """
        Construct metric results with statistics, scenario, and time series
        :param metric_statistics: Metric statistics
        :param scenario: Scenario running this metric to compute a metric score
        :param time_series: Time series object.
        """
        score = self.compute_score(scenario=scenario, metric_statistics=metric_statistics, time_series=time_series)

        result = MetricStatistics(
            metric_computator=self.name,
            name=self.name,
            statistics=metric_statistics,
            time_series=time_series,
            metric_category=self.category,
            metric_score=score,
        )

        return [result]
