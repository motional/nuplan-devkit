from typing import List, Optional

import numpy as np
import numpy.typing as npt

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error_within_bound import (
    PlannerExpertAverageL2ErrorStatistics,
)
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class PlannerMissRateStatistics(MetricBase):
    """Miss rate defined based on the maximum L2 error of planned ego pose w.r.t expert."""

    def __init__(
        self,
        name: str,
        category: str,
        planner_expert_average_l2_error_within_bound_metric: PlannerExpertAverageL2ErrorStatistics,
        max_displacement_threshold: List[float],
        max_miss_rate_threshold: float,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initialize the PlannerMissRateStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param planner_expert_average_l2_error_within_bound_metric: planner_expert_average_l2_error_within_bound metric for each horizon.
        :param max_displacement_threshold: A List of thresholds at different horizons
        :param max_miss_rate_threshold: maximum acceptable miss rate threshold.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._max_displacement_threshold = max_displacement_threshold
        self._max_miss_rate_threshold = max_miss_rate_threshold

        # Initialize lower level metrics
        self._planner_expert_average_l2_error_within_bound_metric = planner_expert_average_l2_error_within_bound_metric

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[-1].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        maximum_displacement_errors = (
            self._planner_expert_average_l2_error_within_bound_metric.maximum_displacement_errors
        )
        comparison_horizon = self._planner_expert_average_l2_error_within_bound_metric.comparison_horizon
        miss_rates: npt.NDArray[np.float64] = np.array(
            [
                np.mean(maximum_displacement_errors[i] > self._max_displacement_threshold[i])
                for i in range(len(comparison_horizon))
            ]
        )

        metric_statistics = [
            Statistic(
                name=f"planner_miss_rate_horizon_{comparison_horizon[ind]}",
                unit=MetricStatisticsType.RATIO.unit,
                value=miss_rate,
                type=MetricStatisticsType.RATIO,
            )
            for ind, miss_rate in enumerate(miss_rates)
        ]
        metric_statistics.append(
            Statistic(
                name=f"{self.name}",
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=float(np.all(miss_rates <= self._max_miss_rate_threshold)),
                type=MetricStatisticsType.BOOLEAN,
            )
        )
        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=metric_statistics, scenario=scenario, metric_score_unit=self.metric_score_unit
        )

        return results
