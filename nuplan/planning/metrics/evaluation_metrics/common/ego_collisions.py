from dataclasses import dataclass
from typing import Dict, List, Set

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricViolation
from nuplan.planning.metrics.utils.metric_violation_aggregator import aggregate_metric_violations
from nuplan.planning.metrics.utils.state_extractors import ego_delta_v_collision
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import box3d_to_polygon


@dataclass
class CollisionData:
    """ Class to retain information about a collision"""
    # ego_collision_side: int  TODO: Report this in the metric
    collision_ego_delta_v: float


@dataclass
class Collisions:
    """ Class to retain information about the collisions at a particular timestamp. """
    timestamp: float  # The timestamp to which the object refers to.
    collision_data: Dict[str, CollisionData]  # Contains the ids of the agents in collision with their collision data


class CollisionExtractor:
    def __init__(self, history: SimulationHistory, metric_name: str, category: str, statistics_name: str):
        self.history = history
        self.collisions: List[Collisions] = []
        self.open_violations: Dict[str, MetricViolation] = {}
        self.violations: List[MetricViolation] = []

        self.metric_name = metric_name
        self.category = category
        self.statistics_name = statistics_name

    def extract_metric(self) -> None:
        """ Extracts the collision violation from the history of Ego and Detection poses. """
        # The IDs of collision on the last timestamp
        for sample in self.history.data:
            ego_state = sample.ego_state
            observation = sample.observation
            timestamp = sample.iteration.time_us

            if not isinstance(observation, Detections):
                raise RuntimeError("cannot compute metric on point cloud")

            collisions = self._find_collisions(ego_state, observation, sample.iteration.time_us)
            in_collision_last = set(self.open_violations.keys())
            new_collisions = {id_: collision_data for id_, collision_data in collisions.collision_data.items() if
                              id_ not in in_collision_last}
            self.start_violations(new_collisions, timestamp)

            ended_collisions = in_collision_last.difference(set(collisions.collision_data.keys()))
            self.end_violations(ended_collisions, timestamp)
        # End all violations
        self.end_violations(set(self.open_violations.keys()), self.history.data[-1].iteration.time_us)

    def start_violations(self, collisions: Dict[str, CollisionData], timestamp: int) -> None:
        """
        Opens the violation window of the given IDs, as they now starting to violate the metric

        :param collisions: The current collisions
        :param timestamp: The current timestamp
        """
        for _id, collision_data in collisions.items():
            collision_violation = MetricViolation(metric_computator=self.metric_name,
                                                  name=self.statistics_name,
                                                  metric_category=self.category,
                                                  unit="m/s",
                                                  start_timestamp=timestamp,
                                                  duration=1,
                                                  extremum=collision_data.collision_ego_delta_v,
                                                  mean=collision_data.collision_ego_delta_v)

            self.open_violations[_id] = collision_violation

    def end_violations(self, ids: Set[str], timestamp: int) -> None:
        """
        Closes the violation window of the given IDs, as they are not violating the metric anymore

        :param ids: The ids which are not violating anymore
        :param timestamp: The current timestamp
        """
        for _id in ids:
            try:
                violation = self.open_violations.pop(_id)

            except KeyError:
                raise KeyError("No open violation found with the given ID!")
            violation.duration = timestamp - violation.start_timestamp
            self.violations.append(violation)

    @staticmethod
    def _find_collisions(ego_state: EgoState, observation: Detections, timestamp: float) -> Collisions:
        """
        Computes the set of IDs of the agents with which ego is colliding.

        :param ego_state: The current state of Ego
        :param observation: The detections at the current time step
        :param timestamp: The current timestamp
        :return: An object containing the IDs of agents with which ego is colliding with.
        """
        # This method can be further optimized by GPU implementation
        collisions = Collisions(timestamp=timestamp, collision_data={})

        for box in observation.boxes:
            agent_polygon = box3d_to_polygon(box)
            if ego_state.car_footprint.oriented_box.geometry.intersects(agent_polygon):
                collision_delta_v = ego_delta_v_collision(ego_state, box)
                collisions.collision_data[box.token] = CollisionData(collision_delta_v)

        return collisions


class EgoCollisionStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Statistics on number of collisions of ego.

        A collision is defined as the event of ego intersecting another bounding box. If the same collision lasts for
        multiple frames, it still counts as a single one.
        :param name: Metric name.
        :param category: Metric category.
        """

        self._name = name
        self._category = category
        self._statistics_name = "ego_collision_statistics"

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

        extractor = CollisionExtractor(history=history, metric_name=self._name, category=self._category,
                                       statistics_name=self._statistics_name)

        extractor.extract_metric()
        violation_statistics = aggregate_metric_violations(extractor.violations, self._name, self._category,
                                                           self._statistics_name)

        return [violation_statistics]
