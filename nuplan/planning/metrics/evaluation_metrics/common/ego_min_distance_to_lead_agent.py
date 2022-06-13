from typing import Dict, List, Optional, Set

import numpy as np

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_at_fault_collisions import (
    Collisions,
    EgoAtFaultCollisionStatistics,
)
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.utils import get_closest_agent_in_position, is_agent_ahead


def get_min_distance_to_lead_agents(
    history: SimulationHistory, all_collisions: List[Collisions], lateral_distance_threshold: float
) -> List[float]:
    """
    Get minimum distance to lead agents in the history
    :param history: History from a simulation engine.
    :param all_collisions: List of all collisions in the history
    :param lateral_distance_threshold: Agents laterally further away than this threshold are not considered
    :return minimum distance to lead agents (pedestrian, vehicle, cyclist, generic objects).
    """
    collided_track_ids: Set[str] = set()
    min_distance_to_lead_agents: List[float] = [np.inf] * len(history)
    timestamps_in_collision = [collision.timestamp for collision in all_collisions]

    for ind, sample in enumerate(history.data):
        ego_state = sample.ego_state
        observation = sample.observation
        timestamp = ego_state.time_point.time_us

        if timestamp in timestamps_in_collision:
            new_collided_tracks = [
                list(collision.collisions_id_data.keys())
                for collision in all_collisions
                if collision.timestamp == timestamp
            ][0]
            collided_track_ids = collided_track_ids.union(set(new_collided_tracks))

        # Update the set of too close lead agents if not (so far) in collision with ego
        closest_agent, closest_distance = get_closest_agent_in_position(
            ego_state, observation, is_agent_ahead, collided_track_ids, lateral_distance_threshold
        )

        min_distance_to_lead_agents[ind] = closest_distance

    return min_distance_to_lead_agents


class EgoMinDistanceToLeadAgent(MetricBase):
    """Ego minimum distance to lead agents metric."""

    def __init__(
        self,
        name: str,
        category: str,
        ego_at_fault_collisions_metric: EgoAtFaultCollisionStatistics,
        min_front_distance: float,
        lateral_distance_threshold: float,
    ):
        """
        Initializes the EgoMinDistanceToLeadAgent class
        :param name: Metric name
        :param category: Metric category
        :param ego_at_fault_collisions_metric: Ego at fault collisions metric
        :param min_front_distance: Minimum acceptable distance  between ego and the front agent
        :param lateral_distance_threshold: Agents laterally further away than this threshold are not considered.
        """
        super().__init__(name=name, category=category)
        self._min_front_distance = min_front_distance
        self._lateral_distance_threshold = lateral_distance_threshold

        self._at_fault_collisions = ego_at_fault_collisions_metric

        # save to load in higher level metrics
        self.results: List[MetricStatistics] = []

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        # Return 1.0 if ego doesn't get too close to a lead agent, 0 otherwise.
        return float(metric_statistics[MetricStatisticsType.BOOLEAN].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        # Extract states of ego from history.
        ego_states = history.extract_ego_state

        # Extract ego timepoints
        time_stamps = extract_ego_time_point(ego_states)

        # Load pre-calculated violations from ego_at_fault_collision metric
        assert (
            self._at_fault_collisions.results
        ), "ego_at_fault_collisions metric must be run prior to calling {}".format(self.name)

        min_distance_to_lead_agents = get_min_distance_to_lead_agents(
            history, self._at_fault_collisions.all_collisions, self._lateral_distance_threshold
        )

        time_series = TimeSeries(unit='meters', time_stamps=list(time_stamps), values=min_distance_to_lead_agents)

        metric_statistics = self._compute_time_series_statistic(time_series=time_series)

        distance_to_lead_agents_within_bound = self._min_front_distance < np.array(min_distance_to_lead_agents)

        metric_statistics[MetricStatisticsType.BOOLEAN] = Statistic(
            name=f'{self.name}_within_bound', unit='boolean', value=bool(np.all(distance_to_lead_agents_within_bound))
        )
        results = self._construct_metric_results(
            metric_statistics=metric_statistics, time_series=time_series, scenario=scenario
        )
        # Save to load in high level metrics
        self.results = results

        return results  # type: ignore
