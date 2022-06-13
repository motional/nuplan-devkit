import statistics
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.base.violation_metric_base import ViolationMetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricViolation, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


@dataclass
class DrivableAreaViolation:
    """
    Class used to keep track of violations, contains the depth of violation as well as their timestamp.
    """

    timestamp: int
    violation_depths: List[float]


class DrivableAreaViolationExtractor:
    """Class to extract depth of driving in non-drivable area."""

    def __init__(self, history: SimulationHistory, metric_name: str, category: str) -> None:
        """
        Initializes the DrivableAreaViolationExtractor class
        :param history: History from a simulation engine
        :param metric_name: Metric name
        :param category: Metric category
        """
        self.history = history
        self.open_violation: Optional[DrivableAreaViolation] = None
        self.violations: List[MetricViolation] = []
        self.distances: List[float] = []
        self.metric_name = metric_name
        self.category = category

    def extract_metric(self) -> None:
        """Extracts the drivable area violations from the history of Ego poses."""
        # Get ego states
        ego_states = self.history.extract_ego_state
        self.distances = [0.0 for _ in range(len(ego_states))]

        # Get frame index for each corner, we get 4 coordinates only since 5th is always the first one
        frame_indices = [
            index
            for index, ego_state in enumerate(ego_states)
            for _ in ego_state.car_footprint.oriented_box.geometry.exterior.coords[:4]
        ]

        # Construct points, get the first 4 coordinates only since the 5th is always the first one
        corner_points = [
            Point2D(*corner)
            for ego_state in ego_states
            for corner in ego_state.car_footprint.oriented_box.geometry.exterior.coords[:4]
        ]

        # Compute if corner points are on the drivable layer
        is_on_layer: List[bool] = [
            self.history.map_api.is_in_layer(corner, layer=SemanticMapLayer.DRIVABLE_AREA) for corner in corner_points
        ]

        drivable_area_violation_indices = np.where([corner_on_layer is False for corner_on_layer in is_on_layer])

        # Unpack and then transpose it from col to row
        transpose_drivable_area_violation_indices = np.transpose(*drivable_area_violation_indices)

        # Get the violation points
        violation_points = [
            corner_points[violation_index] for violation_index in transpose_drivable_area_violation_indices
        ]
        if not violation_points:
            return

        # Compute only distance between violation points and their nearest drivable surfaces
        distances = self.history.map_api.get_distances_matrix_to_nearest_map_object(
            points=violation_points, layer=SemanticMapLayer.DRIVABLE_AREA
        )

        if distances is None:
            return

        # Get max from distances of each ego corner (default: empty list) to their nearest drivable area for each frame
        ego_distances_to_drivable_area: List[float] = [0.0 for _ in range(len(ego_states))]
        for violation_index, distance in zip(transpose_drivable_area_violation_indices, distances):
            status_index = frame_indices[violation_index]
            ego_distances_to_drivable_area[status_index] = max(ego_distances_to_drivable_area[status_index], distance)

        self.distances = ego_distances_to_drivable_area

        for index, distance in enumerate(ego_distances_to_drivable_area):
            sample = self.history.data[index]
            timestamp = sample.iteration.time_us
            if distance > 0:
                violation = DrivableAreaViolation(timestamp, [distance])
            else:
                violation = None

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
        maximal_violation = (
            max(self.open_violation.violation_depths) if higher_is_worse else min(self.open_violation.violation_depths)
        )

        self.violations.append(
            MetricViolation(
                name=self.metric_name,
                metric_computator=self.metric_name,
                metric_category=self.category,
                unit="meters",
                start_timestamp=self.open_violation.timestamp,
                duration=timestamp - self.open_violation.timestamp,
                extremum=maximal_violation,
                mean=statistics.mean(self.open_violation.violation_depths),
            )
        )
        self.open_violation = None


class DrivableAreaViolationStatistics(ViolationMetricBase):
    """Statistics on drivable area violations of ego."""

    def __init__(self, name: str, category: str, max_violation_threshold: int = 0) -> None:
        """
        Initializes the DrivableAreaViolationStatistics class
        :param name: Metric name
        :param category: Metric category
        :param max_violation_threshold: Maximum threshold for the violation.
        """
        super().__init__(name=name, category=category, max_violation_threshold=max_violation_threshold)

        # Save to load in high level metrics
        self.results: List[MetricStatistics] = []

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return: the estimated metric.
        """
        extractor = DrivableAreaViolationExtractor(history=history, metric_name=self._name, category=self._category)

        extractor.extract_metric()

        violation_statistics = self.aggregate_metric_violations(
            metric_violations=extractor.violations, scenario=scenario
        )
        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_series = TimeSeries(unit='meters', time_stamps=list(time_stamps), values=extractor.distances)
        violation_statistics[0].time_series = time_series

        # Save to load in high level metrics
        self.results = violation_statistics
        return violation_statistics  # type: ignore
