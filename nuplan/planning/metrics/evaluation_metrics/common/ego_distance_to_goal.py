import warnings
from typing import Dict, List, Optional

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point, get_ego_distance_to_goal
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoDistanceToGoalStatistics(MetricBase):
    """Ego distance to goal metrics."""

    def __init__(self, name: str, category: str, score_distance_threshold: float) -> None:
        """
        Initializes the EgoDistanceToGoalStatistics class
        :param name: Metric name.
        :param category: Metric category.
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
        return self._compute_distance_metric_score(time_series.values[-1])  # type: ignore

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the distance to goal metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated distance to goal metric.
        """
        goal = history.mission_goal

        if not goal:
            warnings.warn(f'Goal is not specified, skipping {self._name} metric')
            return []

        ego_states = history.extract_ego_state
        distances = get_ego_distance_to_goal(ego_states, goal)

        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_series = TimeSeries(unit='meters', time_stamps=list(time_stamps), values=distances)

        metric_statistics = self._compute_time_series_statistic(time_series=time_series)
        results = self._construct_metric_results(
            metric_statistics=metric_statistics, time_series=time_series, scenario=scenario
        )
        return results  # type: ignore
