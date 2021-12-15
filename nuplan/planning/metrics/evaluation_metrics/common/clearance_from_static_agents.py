from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic
from nuplan.planning.metrics.utils.geometry import signed_lateral_distance, signed_longitudinal_distance
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import box3d_to_polygon


@dataclass
class EgoToAgentDistances:
    """
    Class to keep track of the history of projected distances from ego to an agent.
    It also contains the length of the agent.
    """
    agent_length: float  # Length of agent [m]
    longitudinal_distances: List[float]  # Longitudinal distance from ego to the agent [m]
    lateral_distances: List[float]  # Lateral distance from ego to the agent [m]


class ClearanceFromStaticAgentsStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str, lateral_distance_threshold: float) -> None:
        """
        Metric on clearance while passing static vehicles.
        :param name: Metric name.
        :param category: Metric category.
        :param lateral_distance_threshold: Agents laterally further away than this threshold are not considered
        """

        self._name = name
        self._category = category
        self._lateral_distance_threshold = lateral_distance_threshold
        self._ego_half_length = get_pacifica_parameters().half_length

    @property
    def name(self) -> str:
        """
        Returns the metric name.
        :return: the metric name.
        """

        return self._name

    @property
    def category(self) -> str:
        """
        Returns the metric category.
        :return: the metric category.
        """

        return self._category

    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """

        # Compute projected distances
        agents_distances = self._extract_agent_projected_distances(history)

        clearances_during_passing = self._extract_passing_clearances(agents_distances)

        if not clearances_during_passing:
            return []

        statistics = {MetricStatisticsType.MAX: Statistic(name="max_clearance_overtaking_static_agent", unit="meters",
                                                          value=np.amax(clearances_during_passing)),
                      MetricStatisticsType.MIN: Statistic(name="min_clearance_overtaking_static_agent", unit="meters",
                                                          value=np.amin(clearances_during_passing)),
                      MetricStatisticsType.P90: Statistic(name="p90_clearance_overtaking_static_agent", unit="meters",
                                                          value=np.percentile(np.abs(clearances_during_passing), 90)),
                      }

        result = MetricStatistics(metric_computator=self.name,
                                  name="clearance_from_static_agents_statistics",
                                  statistics=statistics,
                                  time_series=None,
                                  metric_category=self.category)

        return [result]

    def get_overtake_start_idx(self, longitudinal_dist: List[float], idx_overtake: int,
                               critical_dist_abs: float) -> int:
        """
        Finds the index of the element which represents the start of the overtake
        :param longitudinal_dist: longitudinal distances
        :param idx_overtake: index of the distance closest to zero
        :param critical_dist_abs: critical distance which represent start of overtake
        :return: index of the start of overtake
        """
        offset = self._get_overtake_edge(longitudinal_dist[idx_overtake::-1], critical_dist_abs)
        return idx_overtake - offset if offset is not None else 0

    def get_overtake_end_idx(self, longitudinal_dist: List[float], idx_overtake: int, critical_dist_abs: float) -> int:
        """
        Finds the index of the element which represents the end of the overtake
        :param longitudinal_dist: longitudinal distances
        :param idx_overtake: index of the distance closest to zero
        :param critical_dist_abs: critical distance which represent end of overtake
        :return: index of the end of overtake
        """
        offset = self._get_overtake_edge(longitudinal_dist[idx_overtake:], critical_dist_abs)
        return idx_overtake + offset if offset is not None else -1

    @staticmethod
    def _get_overtake_edge(distances: List[float], critical_distance: float) -> Optional[int]:
        """
        Finds the index of the first element which exceeds the given amount in a list
        :param distances: list of distances
        :param critical_distance: threshold distance
        :return: index of the first element exceeding the given amount, None if it doesn't happen
        """
        for idx_start, d in enumerate(distances):
            if abs(d) > critical_distance:
                return idx_start
        return None

    def _extract_agent_projected_distances(self, history: SimulationHistory) -> Dict[str, EgoToAgentDistances]:
        """
        Computes the projected distances, for inactive agents only.

        :param history: The history of the scenario
        :return: A dict containing the projected distances to each inactive track in the entire scenario"""
        agents_distances: Dict[str, EgoToAgentDistances] = {}
        inactive_agents_scenario = self._get_inactive_agents_scenario(history)

        for sample in history.data:
            ego_state = sample.ego_state

            assert isinstance(sample.observation, Detections)
            # TODO: refactor the call below to use agent attributes once available
            inactive_agents = self._get_inactive_agents_sample(sample.observation, inactive_agents_scenario)

            lateral_dist = [signed_lateral_distance(ego_state.center, box3d_to_polygon(inactive_agent)) for
                            inactive_agent in inactive_agents.boxes]
            longitudinal_dist = [signed_longitudinal_distance(ego_state.center, box3d_to_polygon(inactive_agent)) for
                                 inactive_agent in inactive_agents.boxes]

            for agent, lon, lat in zip(inactive_agents.boxes, longitudinal_dist, lateral_dist):
                try:
                    agents_distances[agent.token].longitudinal_distances.append(lon)
                    agents_distances[agent.token].lateral_distances.append(lat)
                except KeyError:
                    agents_distances[agent.token] = EgoToAgentDistances(agent.length, [lon], [lat])

        return agents_distances

    def _extract_passing_clearances(self, agents_distances: Dict[str, EgoToAgentDistances]) -> List[float]:
        """
        Extracts the portion of projected distances relative to the passing of every agent and saves them to a list.

        :param agents_distances: The projected distances to each inactive agent
        :return: A list containing the lateral clearance of all inactive agents while ego is passing them.
        """
        clearances_during_overtake = []
        for distances in agents_distances.values():
            max_longitudinal_dist = max(distances.longitudinal_distances)
            idx_max = distances.longitudinal_distances.index(max_longitudinal_dist)
            min_longitudinal_dist = min(distances.longitudinal_distances)
            idx_min = distances.longitudinal_distances.index(min_longitudinal_dist)

            if max_longitudinal_dist > 0 and min_longitudinal_dist < 0 and idx_max < idx_min:
                overtake_idx = np.argmin(np.abs(distances.longitudinal_distances))
                if abs(distances.lateral_distances[overtake_idx]) < self._lateral_distance_threshold:
                    threshold = self._ego_half_length + distances.agent_length / 2.0
                    start_idx = self.get_overtake_start_idx(distances.longitudinal_distances, overtake_idx,
                                                            threshold)
                    end_idx = self.get_overtake_end_idx(distances.longitudinal_distances, overtake_idx,
                                                        threshold)
                    clearances_during_overtake.extend(np.abs(distances.lateral_distances[start_idx:end_idx + 1]))

        return clearances_during_overtake

    @staticmethod
    def _get_inactive_agents_scenario(history: SimulationHistory) -> Set[str]:
        """
        Get a set of agents which are inactive for the full length of the scenario.
        An inactive agents in this context is an agent that for the entire scenario never moves

        :param history: The history from the scenario
        :return: The set of inactive agents"""
        inactive_agents = set()
        active_agents = set()
        for sample in history.data:
            assert isinstance(sample.observation, Detections)
            for box in sample.observation.boxes:
                inactive_agents.add(box.token) if np.isclose(np.linalg.norm(box.velocity), 0.0) else active_agents.add(
                    box.token)

        return inactive_agents.difference(active_agents)

    @staticmethod
    def _get_inactive_agents_sample(agents: Detections, inactive_agents_set: Set[str]) -> Detections:
        """
        Get a set of agents which are inactive for the full length of the scenario.
        An inactive agents in this context is an agent that for the entire scenario never moves

        :param agents: The agents detected in the current sample
        :param inactive_agents_set: The tokens of the agents known to be inactive for the full scenario
        :return: The detections of inactive agents
        """
        return Detections([agent for agent in agents.boxes if agent.token in inactive_agents_set])
