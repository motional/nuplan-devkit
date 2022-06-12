import logging
from typing import Dict, List, Optional

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import calculate_ego_progress_to_goal
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoProgressToGoalStatistics(MetricBase):
    """Ego progress to goal metric."""

    def __init__(self, name: str, category: str, score_distance_threshold: float) -> None:
        """
        Initializes the EgoProgressToGoalStatistics class
        :param name: Metric name
        :param score_distance_threshold: Distance threshold for the score.
        """
        super().__init__(name=name, category=category)
        self._score_distance_threshold = score_distance_threshold

    def _compute_distance_metric_score(self, distance: float) -> float:
        """
        Compute a metric score based on a distance threshold
        :param distance: A distance value
        :return A metric score between 0 and 1.
        """
        return 1.0 if abs(distance) < self._score_distance_threshold else 0.0

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return self._compute_distance_metric_score(metric_statistics[MetricStatisticsType.VALUE].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the ego progress to goal metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return Ego progress to goal statistics.
        """
        goal = history.mission_goal

        if not goal:
            logging.warning(f'Goal is not specified, skipping {self._name} metric')
            return []

        ego_states = history.extract_ego_state
        ego_progress_value = calculate_ego_progress_to_goal(ego_states, goal)
        statistics = {
            MetricStatisticsType.VALUE: Statistic(
                name='ego_progress_to_goal_value', unit='meters', value=ego_progress_value
            )
        }

        results = self._construct_metric_results(metric_statistics=statistics, time_series=None, scenario=scenario)
        return results  # type: ignore
