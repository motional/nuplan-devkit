import logging
from typing import Dict, List, Optional

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.expert_comparisons import calculate_relative_progress_to_goal
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoRelativeProgressToGoalStatistics(MetricBase):
    """Ego relative progress to goal metric."""

    def __init__(self, name: str, category: str, min_relative_progress_rate: float) -> None:
        """
        Initializes the EgoRelativeProgressToGoalStatistics class
        :param name: Metric name
        :param category: Metric category.
        """
        super().__init__(name=name, category=category)
        self._min_relative_progress_rate = min_relative_progress_rate

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[MetricStatisticsType.RATIO].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns ratio of ego's to expert's progress to goal, and a Boolean variable determine if this ratio is
        larer than a threshold (default 0.5)
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return ego's to expert's relative progress to goal statistics.
        """
        goal = history.mission_goal

        if not goal:
            logging.warning(f'Goal is not specified, skipping {self._name} metric')
            return []

        ego_states = history.extract_ego_state
        expert_ego_states = scenario.get_expert_ego_trajectory()
        ego_expert_relative_progress = calculate_relative_progress_to_goal(ego_states, expert_ego_states, goal)
        ego_expert_relative_progress_normalized = max(0, min(1, ego_expert_relative_progress))

        statistics = {
            MetricStatisticsType.RATIO: Statistic(
                name='ego_to_expert_relative_progress_to_goal',
                unit='ratio',
                value=ego_expert_relative_progress_normalized,
            ),
            MetricStatisticsType.BOOLEAN: Statistic(
                name='ego_making_progress_to_goal',
                unit='boolean',
                value=ego_expert_relative_progress >= self._min_relative_progress_rate,
            ),
        }

        results = self._construct_metric_results(metric_statistics=statistics, time_series=None, scenario=scenario)
        return results  # type: ignore
