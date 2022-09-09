import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import GraphEdgeMapObject, Lane
from nuplan.planning.metrics.evaluation_metrics.base.violation_metric_base import ViolationMetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricViolation, Statistic, TimeSeries
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
        self.violation_depths: List[float] = []
        self.metric_name = metric_name
        self.category = category

    def extract_metric(self, ego_route: List[List[GraphEdgeMapObject]]) -> None:
        """Extracts the drivable area violations from the history of Ego poses."""
        timestamp = None
        for sample, curr_ego_route in zip(self.history.data, ego_route):
            ego_state = sample.ego_state
            timestamp = ego_state.time_point.time_us

            # If no lane or lane connector is associated with pose (such as when ego is
            # outside the drivable area), we won't consider speed limit violation
            if not curr_ego_route:
                violation = None
            else:
                violation = self._get_speed_limit_violation(ego_state, timestamp, curr_ego_route)

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
        if timestamp and self.open_violation:
            self.end_violation(timestamp)

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
                name='speed_limit_violation',
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

    @staticmethod
    def _get_speed_limit_violation(
        ego_state: EgoState, timestamp: int, ego_lane_or_laneconnector: List[GraphEdgeMapObject]
    ) -> Optional[GenericViolation]:
        """
        Computes by how much ego is exceeding the speed limit
        :param ego_state: The current state of Ego
        :param timestamp: The current timestamp
        :return: By how much ego is exceeding the speed limit, none if not violation is present or unable to find
        the speed limit.
        """
        if isinstance(ego_lane_or_laneconnector[0], Lane):
            assert len(ego_lane_or_laneconnector) == 1, 'Ego should can assigned to one lane only'
            speed_limits = [ego_lane_or_laneconnector[0].speed_limit_mps]
        else:
            speed_limits = []
            for map_obj in ego_lane_or_laneconnector:
                edges = map_obj.outgoing_edges + map_obj.incoming_edges
                speed_limits.extend([lane.speed_limit_mps for lane in edges])

        # new map can potentially return None if the GPKG does not contain speed limit data,
        # make sure speed limits exist
        if all(speed_limits):
            max_speed_limit = max(speed_limits)
            exceeding_speed = ego_state.dynamic_car_state.speed - max_speed_limit
            return GenericViolation(timestamp, violation_depths=[exceeding_speed]) if exceeding_speed > 0 else None

        return None


class SpeedLimitComplianceStatistics(ViolationMetricBase):
    """Statistics on speed limit compliance of ego."""

    def __init__(
        self,
        name: str,
        category: str,
        lane_change_metric: EgoLaneChangeStatistics,
        max_violation_threshold: int,
        max_overspeed_value_threshold: float,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initializes the SpeedLimitComplianceStatistics class
        :param name: Metric name
        :param category: Metric category
        :param lane_change_metric: lane change metric
        :param max_violation_threshold: Maximum threshold for the number of violation
        :param max_overspeed_value_threshold: A threshold for overspeed value driving above which is considered more
        dangerous.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(
            name=name,
            category=category,
            max_violation_threshold=max_violation_threshold,
            metric_score_unit=metric_score_unit,
        )
        self._max_overspeed_value_threshold = max_overspeed_value_threshold
        self._lane_change_metric = lane_change_metric

    def _compute_violation_metric_score(self, time_series: TimeSeries) -> float:
        """
        Compute a metric score based on the durtaion and magnitude of the violation compared to the scenario
        duration and a threshold for overspeed value.
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
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        if metric_statistics[-1].value:
            return 1.0

        return float(self._compute_violation_metric_score(time_series=time_series))

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        ego_route = self._lane_change_metric.ego_driven_route
        extractor = SpeedLimitViolationExtractor(history=history, metric_name=self._name, category=self._category)

        extractor.extract_metric(ego_route=ego_route)

        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_series = TimeSeries(
            unit='over_speeding[meters_per_second]', time_stamps=list(time_stamps), values=extractor.violation_depths
        )
        violation_statistics: List[MetricStatistics] = self.aggregate_metric_violations(
            metric_violations=extractor.violations, scenario=scenario, time_series=time_series
        )

        return violation_statistics
