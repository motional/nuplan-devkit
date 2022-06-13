from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.compute import signed_lateral_distance, signed_longitudinal_distance
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


@dataclass
class EgoAgentPair:
    """Class to pair ego and agent."""

    ego_state: EgoState  # Ego state
    agent: Agent  # Agent


@dataclass
class EgoToAgentDistances:
    """
    Class to keep track of the history of projected distances from ego to an agent.
    It also contains the length of the agent.
    """

    agent_lengths: List[float]  # A list of Length of agents [m]
    longitudinal_distances: List[float]  # Longitudinal distance from ego to the agent [m]
    lateral_distances: List[float]  # Lateral distance from ego to the agent [m]


class ClearanceFromStaticAgentsStatistics(MetricBase):
    """Metric on clearance while passing static vehicles."""

    def __init__(self, name: str, category: str, lateral_distance_threshold: float) -> None:
        """
        Initializes the ClearanceFromStaticAgentsStatistics class
        :param name: Metric name
        :param category: Metric category
        :param lateral_distance_threshold: Agents laterally further away than this threshold are not considered.
        """
        super().__init__(name=name, category=category)
        self._lateral_distance_threshold = lateral_distance_threshold
        self._ego_half_length = get_pacifica_parameters().half_length

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        # TODO: Define the metric score
        return 0.0

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        # Compute projected distances
        agents_distances = self._extract_agent_projected_distances(history)

        clearances_during_passing = self._extract_passing_clearances(agents_distances)

        if not clearances_during_passing:
            return []

        statistics = {
            MetricStatisticsType.MAX: Statistic(
                name='max_clearance_overtaking_static_agent', unit='meters', value=np.amax(clearances_during_passing)
            ),
            MetricStatisticsType.MIN: Statistic(
                name='min_clearance_overtaking_static_agent', unit='meters', value=np.amin(clearances_during_passing)
            ),
            MetricStatisticsType.P90: Statistic(
                name='p90_clearance_overtaking_static_agent',
                unit='meters',
                value=np.percentile(np.abs(clearances_during_passing), 90),
            ),
        }

        results = self._construct_metric_results(metric_statistics=statistics, time_series=None, scenario=scenario)
        return results  # type: ignore

    def get_overtake_start_idx(
        self, longitudinal_dist: List[float], idx_overtake: int, critical_dist_abs: float
    ) -> int:
        """
        Finds the index of the element which represents the start of the overtake
        :param longitudinal_dist: longitudinal distances
        :param idx_overtake: index of the distance closest to zero
        :param critical_dist_abs: critical distance which represent start of overtake
        :return index of the start of overtake.
        """
        offset = self._get_overtake_edge(longitudinal_dist[idx_overtake::-1], critical_dist_abs)
        return idx_overtake - offset if offset is not None else 0

    def get_overtake_end_idx(self, longitudinal_dist: List[float], idx_overtake: int, critical_dist_abs: float) -> int:
        """
        Finds the index of the element which represents the end of the overtake
        :param longitudinal_dist: longitudinal distances
        :param idx_overtake: index of the distance closest to zero
        :param critical_dist_abs: critical distance which represent end of overtake
        :return index of the end of overtake.
        """
        offset = self._get_overtake_edge(longitudinal_dist[idx_overtake:], critical_dist_abs)
        return idx_overtake + offset if offset is not None else -1

    @staticmethod
    def _get_overtake_edge(distances: List[float], critical_distance: float) -> Optional[int]:
        """
        Finds the index of the first element which exceeds the given amount in a list
        :param distances: list of distances
        :param critical_distance: threshold distance
        :return index of the first element exceeding the given amount, None if it doesn't happen.
        """
        for idx_start, d in enumerate(distances):
            if abs(d) > critical_distance:
                return idx_start
        return None

    def _extract_agent_projected_distances(self, history: SimulationHistory) -> Dict[str, EgoToAgentDistances]:
        """
        Computes the projected distances, for inactive agents only
        :param history: The history of the scenario
        :return A dict containing the projected distances to each inactive track in the entire scenario.
        """
        agents_distances: Dict[str, EgoToAgentDistances] = {}
        inactive_agents_scenario = self._get_inactive_agents_scenario(history)

        for track_token, ego_agent_pairs in inactive_agents_scenario.items():
            lateral_dist = [
                signed_lateral_distance(ego_agent_pair.ego_state.rear_axle, ego_agent_pair.agent.box.geometry)
                for ego_agent_pair in ego_agent_pairs
            ]
            longitudinal_dist = [
                signed_longitudinal_distance(ego_agent_pair.ego_state.rear_axle, ego_agent_pair.agent.box.geometry)
                for ego_agent_pair in ego_agent_pairs
            ]
            lengths = [ego_agent_pair.agent.box.length for ego_agent_pair in ego_agent_pairs]
            agents_distances[track_token] = EgoToAgentDistances(
                agent_lengths=lengths, longitudinal_distances=longitudinal_dist, lateral_distances=lateral_dist
            )

        return agents_distances

    def _extract_passing_clearances(self, agents_distances: Dict[str, EgoToAgentDistances]) -> List[float]:
        """
        Extracts the portion of projected distances relative to the passing of every agent and saves them to a list
        :param agents_distances: The projected distances to each inactive agent
        :return A list containing the lateral clearance of all inactive agents while ego is passing them.
        """
        clearances_during_overtake = []
        for distances in agents_distances.values():
            max_longitudinal_dist = max(distances.longitudinal_distances)
            idx_max = distances.longitudinal_distances.index(max_longitudinal_dist)
            min_longitudinal_dist = min(distances.longitudinal_distances)
            idx_min = distances.longitudinal_distances.index(min_longitudinal_dist)

            if max_longitudinal_dist > 0 > min_longitudinal_dist and idx_max < idx_min:
                overtake_idx = int(np.argmin(np.abs(distances.longitudinal_distances)))
                if abs(distances.lateral_distances[overtake_idx]) < self._lateral_distance_threshold:
                    threshold = self._ego_half_length + distances.agent_lengths[overtake_idx] / 2.0
                    start_idx = self.get_overtake_start_idx(
                        distances.longitudinal_distances, int(overtake_idx), threshold
                    )
                    end_idx = self.get_overtake_end_idx(distances.longitudinal_distances, int(overtake_idx), threshold)
                    clearances_during_overtake.extend(np.abs(distances.lateral_distances[start_idx : end_idx + 1]))

        return clearances_during_overtake

    @staticmethod
    def _get_inactive_agents_scenario(history: SimulationHistory) -> Dict[str, List[EgoAgentPair]]:
        """
        Get a set of agents which are inactive for the full length of the scenario
        An inactive agents in this context is an agent that for the entire scenario never moves
        :param history: The history from the scenario
        :return A dict of inactive tracks and their ego poses with agents.
        """
        # Collect a series of agents to their tracks
        agent_tracks = defaultdict(list)
        for sample in history.data:
            ego_state = sample.ego_state
            if not isinstance(sample.observation, DetectionsTracks):
                continue
            for tracked_object in sample.observation.tracked_objects.get_agents():
                agent_tracks[tracked_object.track_token].append(EgoAgentPair(ego_state=ego_state, agent=tracked_object))

        inactive_track_agents = defaultdict(list)
        for track_token, ego_agent_pairs in agent_tracks.items():
            velocities: npt.NDArray[np.float64] = np.asarray(
                [ego_agent_pair.agent.velocity.magnitude() for ego_agent_pair in ego_agent_pairs]
            )
            inactive_status = np.isclose(velocities, 0.0)

            # Must all inactive
            if np.sum(inactive_status) != len(velocities):
                continue

            inactive_track_agents[track_token] = ego_agent_pairs

        return inactive_track_agents
