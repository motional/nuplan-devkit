import logging
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import GraphEdgeMapObject, Lane, LaneConnector
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.base.violation_metric_base import ViolationMetricBase
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_violation import DrivableAreaViolationStatistics
from nuplan.planning.metrics.metric_result import (
    MetricStatistics,
    MetricStatisticsType,
    MetricViolation,
    Statistic,
    TimeSeries,
)
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory

logger = logging.getLogger(__name__)


@dataclass
class GenericViolation:
    """Class used to keep track of violations, contains the depth of violation as well as their timestamp."""

    timestamp: int
    violation_depths: List[float]


class SpeedLimitViolationExtractor:
    """Class to extract speed limit violations."""

    def __init__(self, history: SimulationHistory, metric_name: str, category: str) -> None:
        """
        Initializes the SpeedLimitViolationExtractor class
        :param history: History from a simulation engine
        :param metric_name: Metric name
        :param category: Metric category.
        """
        self.history = history
        self.open_violation: Optional[GenericViolation] = None
        self.violations: List[MetricViolation] = []
        self.last_map_object: Optional[GraphEdgeMapObject] = None
        self.violation_depths: List[float] = []
        self.metric_name = metric_name
        self.category = category

    def extract_metric(self, ego_distances_to_drivable_area: List[float]) -> None:
        """Extracts the drivable area violations from the history of Ego poses."""
        for sample, ego_distance_to_drivable_area in zip(self.history.data, ego_distances_to_drivable_area):
            ego_state = sample.ego_state
            timestamp = sample.iteration.time_us

            if ego_distance_to_drivable_area > 0:
                violation = None
            else:
                violation = self._get_speed_limit_violation(ego_state, sample.iteration.time_us)

            if violation:
                if not self.open_violation:
                    self.start_violation(violation)
                else:
                    self.update_violation(violation)
                self.violation_depths.append(violation.violation_depths[0])
            else:
                self.violation_depths.append(0)
                if self.open_violation:
                    self.end_violation(timestamp, higher_is_worse=True)
        # End all violations
        if self.open_violation:
            self.end_violation(self.history.data[-1].iteration.time_us)

    def start_violation(self, violation: GenericViolation) -> None:
        """
        Opens the violation window of the given IDs, as they now starting to violate the metric
        :param violation: The current violation.
        """
        self.open_violation = violation

    def update_violation(self, violation: GenericViolation) -> None:
        """
        Updates the violation if the maximum depth of violation is greater than the current maximum
        :param violation: The current violation.
        """
        assert isinstance(self.open_violation, GenericViolation), 'There is no open violation, cannot update it!'
        self.open_violation.violation_depths.extend(violation.violation_depths)

    def end_violation(self, timestamp: int, higher_is_worse: bool = True) -> None:
        """
        Closes the violation window, as Ego re-enters the non-violating regime
        :param timestamp: The current timestamp
        :param higher_is_worse: True if the violation gravity is monotonic increasing with violation depth.
        """
        assert isinstance(self.open_violation, GenericViolation), 'There is no open violation, cannot end it!'
        maximal_violation = (
            max(self.open_violation.violation_depths) if higher_is_worse else min(self.open_violation.violation_depths)
        )

        self.violations.append(
            MetricViolation(
                name=self.metric_name,
                metric_computator=self.metric_name,
                metric_category=self.category,
                unit='meters_per_second',
                start_timestamp=self.open_violation.timestamp,
                duration=timestamp - self.open_violation.timestamp,
                extremum=maximal_violation,
                mean=statistics.mean(self.open_violation.violation_depths),
            )
        )
        self.open_violation = None

    def _get_speed_limit_no_prior(self, ego_state: EgoState) -> Optional[Union[Lane, LaneConnector]]:
        """
        Gets the current lane or lane connector, along with its speed limit
        :param ego_state: State of ego
        :return: An object with the current map element and speed limit, None if none is found.
        """
        if self.history.map_api.is_in_layer(ego_state.center, SemanticMapLayer.LANE):
            layer = SemanticMapLayer.LANE
        elif self.history.map_api.is_in_layer(ego_state.center, SemanticMapLayer.INTERSECTION):
            layer = SemanticMapLayer.LANE_CONNECTOR
        else:
            return None

        segments: List[Union[Lane, LaneConnector]] = self.history.map_api.get_all_map_objects(ego_state.center, layer)

        # There are areas in intersections that are not covered by lane_connectors, if ego ends up in there, use previous speed limit
        # This will not resolve the issue if ego starts in those areas, hence it's better to assign speed limits to intersections in future
        if not len(segments) and self.last_map_object:
            segments = [self.last_map_object]

        if len(segments):
            segment = segments[0]
            return segment
        else:
            return None

    def _get_speed_limit_with_prior(self, ego_state: EgoState) -> Optional[Union[Lane, LaneConnector]]:
        """
        Gets the current lane or lane connector, along with its speed limit, using an initial guess of where ego is
        :param ego_state: State of ego
        :return: An object with the current map element and speed limit, None if none is found.
        """
        assert isinstance(self.last_map_object, GraphEdgeMapObject)

        # If we are in the same lane or lane connector, nothing to do
        if self.last_map_object.contains_point(ego_state.center):
            return self.last_map_object

        # We check if the upcoming map elements contain the point
        segments = self.last_map_object.outgoing_edges()
        for segment in segments:
            if segment.contains_point(ego_state.center):
                return segment

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
        if self.last_map_object:
            self.last_map_object = self._get_speed_limit_with_prior(ego_state)
        else:
            self.last_map_object = self._get_speed_limit_no_prior(ego_state)

        if self.last_map_object is not None:
            exceeding_speed = ego_state.dynamic_car_state.speed - self.last_map_object.speed_limit_mps
            return GenericViolation(timestamp, violation_depths=[exceeding_speed]) if exceeding_speed > 0 else None

        return None


class SpeedLimitViolationStatistics(ViolationMetricBase):
    """Statistics on speed limit violations of ego."""

    def __init__(
        self,
        name: str,
        category: str,
        drivable_area_violation_metric: DrivableAreaViolationStatistics,
        max_violation_threshold: int,
        max_overspeed_value_threshold: float,
    ) -> None:
        """
        Initializes the SpeedLimitViolationStatistics class
        :param name: Metric name
        :param category: Metric category
        :param drivable_area_violation_metric: drivable area violation metric
        :param max_violation_threshold: Maximum threshold for the number of violation
        :param max_overspeed_value_threshold: A threshold for overspeed value driving above which is considered more dangerous.
        """
        super().__init__(name=name, category=category, max_violation_threshold=max_violation_threshold)
        self._max_overspeed_value_threshold = max_overspeed_value_threshold
        self._drivable_area_violation = drivable_area_violation_metric

    def _compute_violation_metric_score(self, time_series: TimeSeries) -> float:
        """
        Compute a metric score based on the durtaion and magnitude of the violation compared to the scenario duration and a threshold for overspeed value
        :param time_series: A time series for the overspeed
        :return: A metric score between 0 and 1.
        """
        dt_in_sec = np.mean(np.diff(time_series.time_stamps)) * 1e-6
        scenario_duration_in_sec = (time_series.time_stamps[-1] - time_series.time_stamps[0]) * 1e-6
        if scenario_duration_in_sec <= 0:
            logger.warning('Scenario duration is 0 or less!')
            return 1.0
        # Adding a small tolerance to handle cases where max_overspeed_value_threshold is specified as 0
        max_overspeed_value_threshold = max(self._max_overspeed_value_threshold, 1e-3)
        violation_loss = (
            np.sum(time_series.values) * dt_in_sec / (max_overspeed_value_threshold * scenario_duration_in_sec)
        )
        return float(max(0.0, 1.0 - violation_loss))

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: TimeSeries,
    ) -> float:
        """Inherited, see superclass."""
        if metric_statistics[MetricStatisticsType.BOOLEAN].value:
            return 1.0

        return float(self._compute_violation_metric_score(time_series=time_series))

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        ego_distances_to_drivable_area = self._drivable_area_violation.results[0].time_series.values
        extractor = SpeedLimitViolationExtractor(history=history, metric_name=self._name, category=self._category)

        extractor.extract_metric(ego_distances_to_drivable_area=ego_distances_to_drivable_area)

        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_series = TimeSeries(unit='mps', time_stamps=list(time_stamps), values=extractor.violation_depths)
        violation_statistics = self.aggregate_metric_violations(
            metric_violations=extractor.violations, scenario=scenario, time_series=time_series
        )

        return violation_statistics  # type: ignore
