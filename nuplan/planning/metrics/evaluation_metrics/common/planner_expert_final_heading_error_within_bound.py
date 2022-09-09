from typing import List, Optional

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error_within_bound import (
    PlannerExpertAverageL2ErrorStatistics,
)
from nuplan.planning.metrics.metric_result import MetricStatistics, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class PlannerExpertFinalHeadingErrorStatistics(MetricBase):
    """
    Absolute difference between planned ego heading and expert heading at the final pose given a comparison time horizon.
    """

    def __init__(
        self,
        name: str,
        category: str,
        planner_expert_average_l2_error_within_bound_metric: PlannerExpertAverageL2ErrorStatistics,
        max_final_heading_error_threshold: float,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initialize the PlannerExpertFinalHeadingErrorStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param planner_expert_average_l2_error_within_bound_metric: planner_expert_average_l2_error_within_bound metric.
        :param max_final_heading_error_threshold: Maximum acceptable error threshold.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)

        # Initialize lower level metrics
        self._planner_expert_average_l2_error_within_bound_metric = planner_expert_average_l2_error_within_bound_metric
        self._max_final_heading_error_threshold = max_final_heading_error_threshold

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(max(0, 1 - metric_statistics[-1].value / self._max_final_heading_error_threshold))

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        final_heading_errors = self._planner_expert_average_l2_error_within_bound_metric.final_heading_errors
        ego_timestamps_sampled = self._planner_expert_average_l2_error_within_bound_metric.ego_timestamps_sampled
        selected_frames = self._planner_expert_average_l2_error_within_bound_metric.selected_frames
        comparison_horizon = self._planner_expert_average_l2_error_within_bound_metric.comparison_horizon

        results: List[MetricStatistics] = self._construct_open_loop_metric_results(
            scenario,
            comparison_horizon,
            self._max_final_heading_error_threshold,
            metric_values=final_heading_errors,
            name='planner_expert_FHE',
            unit='radian',
            timestamps_sampled=ego_timestamps_sampled,
            metric_score_unit=self.metric_score_unit,
            selected_frames=selected_frames,
        )

        return results
