from typing import List, Optional

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error import (
    PlannerExpertAverageL2ErrorStatistics,
)
from nuplan.planning.metrics.metric_result import MetricStatistics, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class PlannerExpertAverageHeadingErrorStatistics(MetricBase):
    """
    L2 error of planned ego pose w.r.t expert at the final pose (i.e., final pose given the
    planner time horizon or final pose of expert whichever is smaller).
    """

    def __init__(
        self,
        name: str,
        category: str,
        planner_expert_average_l2_error_metric: PlannerExpertAverageL2ErrorStatistics,
    ) -> None:
        """
        Initialize the PlannerExpertFinalL2ErrorStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param planner_expert_average_l2_error_metric: planner_expert_average_l2_error metric.
        """
        super().__init__(name=name, category=category)

        # Initialize lower level metrics
        self._planner_expert_average_l2_error_metric = planner_expert_average_l2_error_metric

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
        average_heading_errors = self._planner_expert_average_l2_error_metric.average_heading_errors
        ego_timestamps_sampled = self._planner_expert_average_l2_error_metric.ego_timestamps_sampled

        time_series = TimeSeries(
            unit='radian',
            time_stamps=ego_timestamps_sampled[: len(average_heading_errors)],
            values=average_heading_errors,
        )

        metric_statistics = self._compute_time_series_statistic(time_series=time_series)

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=metric_statistics, scenario=scenario, time_series=time_series
        )
        return results
