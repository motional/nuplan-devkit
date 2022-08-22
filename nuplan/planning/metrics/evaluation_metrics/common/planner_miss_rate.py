from typing import List, Optional

import numpy as np

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error import (
    PlannerExpertAverageL2ErrorStatistics,
)
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class PlannerMissRateStatistics(MetricBase):
    """Average of the maximum L2 error of planned ego pose w.r.t expert."""

    def __init__(
        self,
        name: str,
        category: str,
        planner_expert_average_l2_error_metric: PlannerExpertAverageL2ErrorStatistics,
        max_displacement_threshold: float,
        max_miss_rate_threshold: float,
    ) -> None:
        """
        Initialize the PlannerMissRateStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param planner_expert_average_l2_error_metric: planner_expert_average_l2_error metric.
        :param max_displacement_threshold: maximum acceptable displacement threshol.
        :param max_miss_rate_threshold: maximum acceptable miss rate threshold.
        """
        super().__init__(name=name, category=category)
        self._max_displacement_threshold = max_displacement_threshold
        self._max_miss_rate_threshold = max_miss_rate_threshold

        # Initialize lower level metrics
        self._planner_expert_average_l2_error_metric = planner_expert_average_l2_error_metric

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        maximum_displacement_errors = self._planner_expert_average_l2_error_metric.maximum_displacement_errors
        ego_timestamps_sampled = self._planner_expert_average_l2_error_metric.ego_timestamps_sampled
        miss_rate = np.mean(np.array(maximum_displacement_errors) > self._max_displacement_threshold)
        miss_rate_is_below_threshold = miss_rate <= self._max_miss_rate_threshold
        statistics = [
            Statistic(
                name='planner_miss_rate_ratio',
                unit=MetricStatisticsType.RATIO.unit,
                value=float(miss_rate),
                type=MetricStatisticsType.RATIO,
            ),
            Statistic(
                name='planner_miss_rate_is_below_threshold',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=miss_rate_is_below_threshold,
                type=MetricStatisticsType.BOOLEAN,
            ),
        ]
        time_series = TimeSeries(
            unit='meters',
            time_stamps=ego_timestamps_sampled[: len(maximum_displacement_errors)],
            values=maximum_displacement_errors,
        )

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=statistics, scenario=scenario, time_series=time_series
        )
        return results
