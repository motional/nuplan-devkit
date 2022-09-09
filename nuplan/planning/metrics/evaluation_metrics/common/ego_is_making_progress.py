from typing import List, Optional

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import (
    EgoProgressAlongExpertRouteStatistics,
)
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoIsMakingProgressStatistics(MetricBase):
    """
    Check if ego trajectory is making progress along expert route more than a minimum required progress.
    """

    def __init__(
        self,
        name: str,
        category: str,
        ego_progress_along_expert_route_metric: EgoProgressAlongExpertRouteStatistics,
        min_progress_threshold: float,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initializes the EgoIsMakingProgressStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_progress_along_expert_route_metric: Ego progress along expert route metric
        :param min_progress_threshold: minimimum required progress threshold
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._min_progress_threshold = min_progress_threshold

        # Initialize lower level metrics
        self._ego_progress_along_expert_route_metric = ego_progress_along_expert_route_metric

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
        Returns the ego_is_making_progress metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        # Load ego_progress_along_expert_route ratio
        ego_is_making_progress = (
            self._ego_progress_along_expert_route_metric.results[0].statistics[-1].value >= self._min_progress_threshold
        )
        statistics = [
            Statistic(
                name='ego_is_making_progress',
                unit='boolean',
                value=ego_is_making_progress,
                type=MetricStatisticsType.BOOLEAN,
            )
        ]

        results = self._construct_metric_results(
            metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit
        )

        return results  # type: ignore
