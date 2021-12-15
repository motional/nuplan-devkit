import statistics
from dataclasses import dataclass
from typing import List, Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricViolation
from nuplan.planning.metrics.utils.metric_violation_aggregator import aggregate_metric_violations
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import ego_state_to_box_3d


@dataclass
class DrivableAreaViolation:
    """ Class used to keep track of violations, contains the depth of violation as well as their timestamp."""
    timestamp: int
    violation_depths: List[float]


class DrivableAreaViolationExtractor:
    def __init__(self, history: SimulationHistory, metric_name: str, category: str, statistics_name: str) -> None:
        self.history = history
        self.open_violation: Optional[DrivableAreaViolation] = None
        self.violations: List[MetricViolation] = []

        self.metric_name = metric_name
        self.category = category
        self.statistics_name = statistics_name

    def extract_metric(self) -> None:
        """ Extracts the drivable area violations from the history of Ego poses. """
        for sample in self.history.data:
            ego_state = sample.ego_state
            timestamp = sample.iteration.time_us

            violation = self._get_drivable_area_violation(ego_state, sample.iteration.time_us)
            if violation:
                if not self.open_violation:
                    self.start_violation(violation)
                else:
                    self.update_violation(violation)
            elif not violation and self.open_violation:
                self.end_violation(timestamp, higher_is_worse=True)
        # End all violations
        if self.open_violation:
            self.end_violation(self.history.data[-1].iteration.time_us)

    def start_violation(self, violation: DrivableAreaViolation) -> None:
        """
        Opens the violation window of the given IDs, as they now starting to violate the metric

        :param violation: The current violation
        """
        self.open_violation = violation

    def update_violation(self, violation: DrivableAreaViolation) -> None:
        """
        Updates the violation if the maximum depth of violation is greater than the current maximum

        :param violation: The current violation
        """
        assert isinstance(self.open_violation, DrivableAreaViolation), "There is no open violation, cannot update it!"
        self.open_violation.violation_depths.extend(violation.violation_depths)

    def end_violation(self, timestamp: int, higher_is_worse: bool = True) -> None:
        """
        Closes the violation window, as Ego re-entered the drivable area

        :param timestamp: The current timestamp
        :param higher_is_worse: True if the violation gravity is monotonic increasing with violation depth

        """
        assert isinstance(self.open_violation, DrivableAreaViolation), "There is no open violation, cannot end it!"
        maximal_violation = max(self.open_violation.violation_depths) if higher_is_worse else min(
            self.open_violation.violation_depths)

        self.violations.append(MetricViolation(name=self.statistics_name,
                                               metric_computator=self.metric_name,
                                               metric_category=self.category,
                                               unit="meters",
                                               start_timestamp=self.open_violation.timestamp,
                                               duration=timestamp - self.open_violation.timestamp,
                                               extremum=maximal_violation,
                                               mean=statistics.mean(self.open_violation.violation_depths)))
        self.open_violation = None

    def _get_drivable_area_violation(self, ego_state: EgoState, timestamp: int) -> \
            Optional[DrivableAreaViolation]:
        """
        Computes by how much ego is outside the drivable area

        :param ego_state: The current state of Ego
        :param timestamp: The current timestamp
        :return: By how fat ego is outside the drivable area, None if completely inside
        """

        ego_box = ego_state_to_box_3d(ego_state)
        corner_points = [Point2D(*corner) for corner in ego_box.bottom_corners[:2].T]
        # Distance to drivable area is zero when the point is in drivable area, otherwise outside
        drivable_area_dist = min([self.dist_to_drivable_surface(point) for point in corner_points])

        return DrivableAreaViolation(timestamp, [drivable_area_dist]) if drivable_area_dist > 0 else None

    def dist_to_drivable_surface(self, point: Point2D) -> float:
        """
        :param point: [m] x, y coordinates in global frame
        :return dist from [x, y] to drivable_surface
        """
        _, distance_to_drivable_area = \
            self.history.map_api.get_distance_to_nearest_map_object(point, SemanticMapLayer.DRIVABLE_AREA)
        return float(distance_to_drivable_area)


class DrivableAreaViolationStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Statistics on drivable area violations of ego.

        :param name: Metric name.
        :param category: Metric category.
        """

        self._name = name
        self._category = category
        self._statistics_name = "drivable_area_violation_statistics"

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

        extractor = DrivableAreaViolationExtractor(history=history, metric_name=self._name, category=self._category,
                                                   statistics_name=self._statistics_name)

        extractor.extract_metric()

        violation_statistics = aggregate_metric_violations(extractor.violations, self._name, self._category,
                                                           self._statistics_name)

        return [violation_statistics]
