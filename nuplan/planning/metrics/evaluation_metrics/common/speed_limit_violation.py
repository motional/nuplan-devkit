import statistics
from dataclasses import dataclass
from typing import List, Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import GraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricViolation
from nuplan.planning.metrics.utils.metric_violation_aggregator import aggregate_metric_violations
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


@dataclass
class GenericViolation:
    """ Class used to keep track of violations, contains the depth of violation as well as their timestamp."""
    timestamp: int
    violation_depths: List[float]


@dataclass
class RoadElementAndSpeedLimit:
    road_element: GraphEdgeMapObject
    speed_limit_mps: float


class SpeedLimitViolationExtractor:
    def __init__(self, history: SimulationHistory, metric_name: str, category: str, statistics_name: str) -> None:
        self.history = history
        self.open_violation: Optional[GenericViolation] = None
        self.violations: List[MetricViolation] = []
        self.last_element_and_speed_limit: Optional[RoadElementAndSpeedLimit] = None

        self.metric_name = metric_name
        self.category = category
        self.statistics_name = statistics_name

    def extract_metric(self) -> None:
        """ Extracts the drivable area violations from the history of Ego poses. """
        for sample in self.history.data:
            ego_state = sample.ego_state
            timestamp = sample.iteration.time_us

            violation = self._get_speed_limit_violation(ego_state, sample.iteration.time_us)
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

    def start_violation(self, violation: GenericViolation) -> None:
        """
        Opens the violation window of the given IDs, as they now starting to violate the metric

        :param violation: The current violation
        """
        self.open_violation = violation

    def update_violation(self, violation: GenericViolation) -> None:
        """
        Updates the violation if the maximum depth of violation is greater than the current maximum

        :param violation: The current violation
        """
        assert isinstance(self.open_violation, GenericViolation), "There is no open violation, cannot update it!"
        self.open_violation.violation_depths.extend(violation.violation_depths)

    def end_violation(self, timestamp: int, higher_is_worse: bool = True) -> None:
        """
        Closes the violation window, as Ego re-enters the non-violating regime

        :param timestamp: The current timestamp
        :param higher_is_worse: True if the violation gravity is monotonic increasing with violation depth

        """
        assert isinstance(self.open_violation, GenericViolation), "There is no open violation, cannot end it!"
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

    def _get_speed_limit_no_prior(self, ego_state: EgoState) -> Optional[RoadElementAndSpeedLimit]:
        """
        Gets the current lane or lane connector, along with its speed limit.
        :param ego_state: State of ego
        :returns: An object with the current map element and speed limit, None if none is found
        """

        if self.history.map_api.is_in_layer(ego_state.center, SemanticMapLayer.LANE):
            layer = SemanticMapLayer.LANE
        elif self.history.map_api.is_in_layer(ego_state.center, SemanticMapLayer.INTERSECTION):
            layer = SemanticMapLayer.LANE_CONNECTOR
        else:
            return None

        segments: List[GraphEdgeMapObject] = self.history.map_api.get_all_map_objects(ego_state.center, layer)
        segment = segments[0]
        return RoadElementAndSpeedLimit(segment, segment.speed_limit_mps)

    def _get_speed_limit_with_prior(self, ego_state: EgoState) -> Optional[RoadElementAndSpeedLimit]:
        """
        Gets the current lane or lane connector, along with its speed limit, using an initial guess of where ego is.
        :param ego_state: State of ego
        :returns: An object with the current map element and speed limit, None if none is found
        """
        assert isinstance(self.last_element_and_speed_limit, RoadElementAndSpeedLimit)

        # If we are in the same lane or lane connector, nothing to do
        if self.last_element_and_speed_limit.road_element.contains_point(ego_state.center):
            return self.last_element_and_speed_limit

        # We check if the upcoming map elements contain the point
        segments = self.last_element_and_speed_limit.road_element.outgoing_edges()
        for segment in segments:
            if segment.contains_point(ego_state.center):
                return RoadElementAndSpeedLimit(segment, segment.speed_limit_mps)

        # If everything else fails we resort to compute from scratch
        return self._get_speed_limit_no_prior(ego_state)

    def _get_speed_limit_violation(self, ego_state: EgoState, timestamp: int) -> Optional[GenericViolation]:
        """
        Computes by how much ego is exceeding the speed limit

        :param ego_state: The current state of Ego
        :param timestamp: The current timestamp
        :return: By how much ego is exceeding the speed limit, none if not violation is present or unable to find
        the speed limit.
        """

        if self.last_element_and_speed_limit:
            self.last_element_and_speed_limit = self._get_speed_limit_with_prior(ego_state)
        else:
            self.last_element_and_speed_limit = self._get_speed_limit_no_prior(ego_state)

        if self.last_element_and_speed_limit is not None:
            exceeding_speed = ego_state.dynamic_car_state.speed - self.last_element_and_speed_limit.speed_limit_mps
            return GenericViolation(timestamp, violation_depths=[exceeding_speed]) if exceeding_speed > 0 else None

        return None


class SpeedLimitViolationStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Statistics on drivable area violations of ego.

        :param name: Metric name.
        :param category: Metric category.
        """

        self._name = name
        self._category = category
        self._statistics_name = "speed_limit_violation_statistics"

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

        extractor = SpeedLimitViolationExtractor(history=history, metric_name=self._name, category=self._category,
                                                 statistics_name=self._statistics_name)

        extractor.extract_metric()

        violation_statistics = aggregate_metric_violations(extractor.violations, self._name, self._category,
                                                           self._statistics_name)

        return [violation_statistics]
