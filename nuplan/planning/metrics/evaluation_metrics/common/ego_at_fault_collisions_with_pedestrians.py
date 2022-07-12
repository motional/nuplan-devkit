from typing import List

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.evaluation_metrics.base.violation_metric_base import ViolationMetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricViolation
from nuplan.planning.metrics.utils.collision_utils import get_fault_type_violation
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoAtFaultCollisionPedestrianStatistics(ViolationMetricBase):
    """
    Statistics on number and energy of at fault collisions of ego with pedestrians.
    A collision is defined as the event of ego intersecting another bounding box. If the same collision lasts for
    multiple frames, it still counts as a single one and the first frame is evaluated for the violation metric.
    """

    def __init__(
        self,
        name: str,
        category: str,
        ego_at_fault_collisions_metric: EgoAtFaultCollisionStatistics,
        max_violation_threshold: int,
    ) -> None:
        """
        Initializes the EgoAtFaultCollisionPedestrianStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_at_fault_collisions_metric: Ego at fault collisions metric computed prior to the current metric
        :param max_violation_threshold: Maximum threshold for the violation.
        """
        super().__init__(name=name, category=category, max_violation_threshold=max_violation_threshold)

        # Initialize lower level metrics
        self._ego_at_fault_collisions_metric = ego_at_fault_collisions_metric

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the collision metric.
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated pedestrian collision energy and counts.
        """
        # Load pre-calculated collisions from ego_at_fault_collision metric
        all_at_fault_collisions = self._ego_at_fault_collisions_metric.all_at_fault_collisions

        all_violations: List[MetricViolation] = get_fault_type_violation(
            [TrackedObjectType.PEDESTRIAN],
            all_at_fault_collisions,
            self.name,
            self.category,
        )

        violation_statistics = self.aggregate_metric_violations(all_violations, scenario=scenario)

        return violation_statistics  # type: ignore
