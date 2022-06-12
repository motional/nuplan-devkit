from dataclasses import dataclass
from typing import Dict, List, Set

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.metrics.evaluation_metrics.base.violation_metric_base import ViolationMetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricViolation
from nuplan.planning.metrics.utils.state_extractors import ego_delta_v_collision
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


@dataclass
class CollisionData:
    """Class to retain information about a collision"""

    collision_ego_delta_v: float


@dataclass
class Collisions:
    """Class to retain information about the collisions at a particular timestamp."""

    timestamp: float  # The timestamp to which the object refers to.
    collision_data: Dict[str, CollisionData]  # Contains the ids of the agents in collision with their collision data


class CollisionExtractor:
    """Class to extract collisions."""

    def __init__(self, history: SimulationHistory, metric_name: str, category: str):
        """
        Initializes the CollisionExtractor class
        :param history: History from a simulation engine.
        :param metric_name: Metric name
        :param category: Metric category
        """
        self.history = history
        self.collisions: List[Collisions] = []
        self.open_violations: Dict[str, MetricViolation] = {}
        self.violations: List[MetricViolation] = []

        self.metric_name = metric_name
        self.category = category

    def extract_metric(self) -> None:
        """Extracts the collision violation from the history of Ego and Detection poses."""
        # The IDs of collision on the last timestamp
        for sample in self.history.data:
            ego_state = sample.ego_state
            observation = sample.observation
            timestamp = sample.iteration.time_us

            if not isinstance(observation, DetectionsTracks):
                raise RuntimeError('cannot compute metric on point cloud')

            collisions = self._find_collisions(ego_state, observation, sample.iteration.time_us)
            in_collision_last = set(self.open_violations.keys())
            new_collisions = {
                id_: collision_data
                for id_, collision_data in collisions.collision_data.items()
                if id_ not in in_collision_last
            }
            self.start_violations(new_collisions, timestamp)

            ended_collisions = in_collision_last.difference(set(collisions.collision_data.keys()))
            self.end_violations(ended_collisions, timestamp)
        # End all violations
        self.end_violations(set(self.open_violations.keys()), self.history.data[-1].iteration.time_us)

    def start_violations(self, collisions: Dict[str, CollisionData], timestamp: int) -> None:
        """
        Opens the violation window of the given IDs, as they now starting to violate the metric
        :param collisions: The current collisions
        :param timestamp: The current timestamp.
        """
        for _id, collision_data in collisions.items():
            collision_violation = MetricViolation(
                metric_computator=self.metric_name,
                name=self.metric_name,
                metric_category=self.category,
                unit='m/s',
                start_timestamp=timestamp,
                duration=1,
                extremum=collision_data.collision_ego_delta_v,
                mean=collision_data.collision_ego_delta_v,
            )

            self.open_violations[_id] = collision_violation

    def end_violations(self, ids: Set[str], timestamp: int) -> None:
        """
        Closes the violation window of the given IDs, as they are not violating the metric anymore
        :param ids: The ids which are not violating anymore
        :param timestamp: The current timestamp.
        """
        for _id in ids:
            try:
                violation = self.open_violations.pop(_id)

            except KeyError:
                raise KeyError('No open violation found with the given ID!')
            violation.duration = timestamp - violation.start_timestamp
            self.violations.append(violation)

    @staticmethod
    def _find_collisions(ego_state: EgoState, observation: DetectionsTracks, timestamp: float) -> Collisions:
        """
        Computes the set of IDs of the agents with which ego is colliding
        :param ego_state: The current state of Ego
        :param observation: The detections at the current time step
        :param timestamp: The current timestamp
        :return An object containing the IDs of agents with which ego is colliding with.
        """
        # This method can be further optimized by GPU implementation
        collisions = Collisions(timestamp=timestamp, collision_data={})

        for tracked_object in observation.tracked_objects:
            if ego_state.car_footprint.oriented_box.geometry.intersects(tracked_object.box.geometry):
                collision_delta_v = ego_delta_v_collision(ego_state, tracked_object)
                collisions.collision_data[tracked_object.track_token] = CollisionData(collision_delta_v)

        return collisions


class EgoCollisionStatistics(ViolationMetricBase):
    """
    Statistics on number of collisions of ego.  A collision is defined as the event of ego intersecting another bounding box.
    If the same collision lasts for multiple frames, it still counts as a single one with the duration be as the length of the continuous frames.
    """

    def __init__(self, name: str, category: str, max_violation_threshold: int) -> None:
        """
        Initializes the EgoCollisionStatistics class
        :param name: Metric name.
        :param category: Metric category.
        :param max_violation_threshold: Maximum threshold for the violation.
        """
        super().__init__(name=name, category=category, max_violation_threshold=max_violation_threshold)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        extractor = CollisionExtractor(history=history, metric_name=self._name, category=self._category)

        extractor.extract_metric()
        violation_statistics = self.aggregate_metric_violations(extractor.violations, scenario=scenario)

        return violation_statistics  # type: ignore
