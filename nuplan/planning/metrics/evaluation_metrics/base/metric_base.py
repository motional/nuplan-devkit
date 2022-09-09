from typing import List, Optional

import numpy as np
import numpy.typing as npt

from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class MetricBase(AbstractMetricBuilder):
    """Base class for evaluation of metrics."""

    def __init__(self, name: str, category: str, metric_score_unit: Optional[str] = None) -> None:
        """
        Initializer for MetricBase
        :param name: Metric name
        :param category: Metric category.
        :param metric_score_unit: Metric final score unit.
        """
        self._name = name
        self._category = category
        self._metric_score_unit = metric_score_unit

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

    @property
    def metric_score_unit(self) -> Optional[str]:
        """
        Returns the metric final score unit.
        """
        return self._metric_score_unit

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
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
        self, time_series: TimeSeries, statistics_type_list: Optional[List[MetricStatisticsType]] = None
    ) -> List[Statistic]:
        """
        Compute metric statistics in time series.
        :param time_series: time series (with float values).
        :param statistics_type_list: List of available types such as [MetricStatisticsType.MAX,
        MetricStatisticsType.MIN, MetricStatisticsType.MEAN, MetricStatisticsType.P90]. Use all if set to None.
        :return A list of metric statistics.
        """
        values = time_series.values
        assert values, "Time series values cannot be empty!"
        unit = time_series.unit

        if statistics_type_list is None:
            statistics_type_list = [
                MetricStatisticsType.MAX,
                MetricStatisticsType.MIN,
                MetricStatisticsType.MEAN,
                MetricStatisticsType.P90,
            ]
        statistics = []
        for statistics_type in statistics_type_list:
            if statistics_type == MetricStatisticsType.MAX:
                name = f"max_{self.name}"
                value = np.nanmax(values)
            elif statistics_type == MetricStatisticsType.MEAN:
                name = f"avg_{self.name}"
                value = np.nanmean(values)
            elif statistics_type == MetricStatisticsType.MIN:
                name = f"min_{self.name}"
                value = np.nanmin(values)
            elif statistics_type == MetricStatisticsType.P90:
                name = f"p90_{self.name}"
                # Use the closest observation to return the actual data instead of linear interpolation
                value = np.nanpercentile(values, 90, method='closest_observation')
            else:
                raise TypeError('Other metric types statistics cannot be created by compute_statistics()')

            statistics.append(Statistic(name=name, unit=unit, value=value, type=statistics_type))

        return statistics

    def _construct_metric_results(
        self,
        metric_statistics: List[Statistic],
        scenario: AbstractScenario,
        metric_score_unit: Optional[str] = None,
        time_series: Optional[TimeSeries] = None,
    ) -> List[MetricStatistics]:
        """
        Construct metric results with statistics, scenario, and time series
        :param metric_statistics: A list of metric statistics
        :param scenario: Scenario running this metric to compute a metric score
        :param metric_score_unit: Unit for the metric final score.
        :param time_series: Time series object.
        :return: A list of metric statistics.
        """
        score = self.compute_score(scenario=scenario, metric_statistics=metric_statistics, time_series=time_series)

        result = MetricStatistics(
            metric_computator=self.name,
            name=self.name,
            statistics=metric_statistics,
            time_series=time_series,
            metric_category=self.category,
            metric_score=score,
            metric_score_unit=metric_score_unit,
        )

        return [result]

    def _construct_open_loop_metric_results(
        self,
        scenario: AbstractScenario,
        comparison_horizon: List[int],
        maximum_threshold: float,
        metric_values: npt.NDArray[np.float64],
        name: str,
        unit: str,
        timestamps_sampled: List[int],
        metric_score_unit: str,
        selected_frames: List[int],
    ) -> List[MetricStatistics]:
        """
        Construct metric results with statistics, scenario, and time series for open_loop metrics.
        :param scenario: Scenario running this metric to compute a metric score.
        :param comparison_horizon: List of horizon times in future (s) to find displacement errors.
        :param maximum_threshold: Maximum acceptable error threshold.
        :param metric_values: Time series object.
        :param name: name of timeseries.
        :param unit: metric unit.
        :param timestamps_sampled:A list of sampled timestamps.
        :param metric_score_unit: Unit for the metric final score.
        :param selected_frames: List sampled indices for nuboard Timeseries frames
        :return: A list of metric statistics.
        """
        metric_statistics: List[Statistic] = [
            Statistic(
                name=f"{name}_horizon_{horizon}",
                unit=unit,
                value=np.mean(metric_values[ind]),
                type=MetricStatisticsType.MEAN,
            )
            for ind, horizon in enumerate(comparison_horizon)
        ]
        metric_statistics.extend(
            [
                Statistic(
                    name=f"{self.name}",
                    unit=MetricStatisticsType.BOOLEAN.unit,
                    value=np.mean(metric_values) <= maximum_threshold,
                    type=MetricStatisticsType.BOOLEAN,
                ),
                Statistic(
                    name=f"avg_{name}_over_all_horizons",
                    unit=unit,
                    value=np.mean(metric_values),
                    type=MetricStatisticsType.MEAN,
                ),
            ]
        )
        metric_values_over_horizons_at_each_time = np.mean(metric_values, axis=0)

        time_series = TimeSeries(
            unit=f"avg_{name}_over_all_horizons [{unit}]",
            time_stamps=timestamps_sampled,
            values=list(metric_values_over_horizons_at_each_time),
            selected_frames=selected_frames,
        )
        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=metric_statistics,
            scenario=scenario,
            metric_score_unit=metric_score_unit,
            time_series=time_series,
        )

        return results
