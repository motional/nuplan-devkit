from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Point
from sympy import Point2D

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import GraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.route_extractor import CornersGraphEdgeMapObject
from nuplan.planning.metrics.utils.state_extractors import extract_ego_corners, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class DrivableAreaComplianceStatistics(MetricBase):
    """Statistics on drivable area compliance of ego."""

    def __init__(
        self,
        name: str,
        category: str,
        lane_change_metric: EgoLaneChangeStatistics,
        max_violation_threshold: float,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initialize the DrivableAreaComplianceStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param lane_change_metric: lane change metric.
        :param max_violation_threshold: [m] tolerance threshold.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)

        # Save to load in high level metrics
        self.results: List[MetricStatistics] = []
        self._lane_change_metric = lane_change_metric
        self._max_violation_threshold = max_violation_threshold

    @staticmethod
    def not_in_drivable_area_with_route_object(
        pose: Point2D, route_object: List[GraphEdgeMapObject], map_api: AbstractMap
    ) -> bool:
        """
        Return a boolean is_in_drivable_area.
        :param pose: pose.
        :param route_object: lane/lane connector of that pose or empty list.
        :param map_api: map.
        :return: a boolean is_in_drivable_area.
        """
        return not route_object and not map_api.is_in_layer(pose, layer=SemanticMapLayer.DRIVABLE_AREA)

    @staticmethod
    def compute_distance_to_map_objects_list(pose: Point2D, map_objects: List[GraphEdgeMapObject]) -> float:
        """
        Compute the min distance to a list of map objects.
        :param pose: pose.
        :param map_objects: list of map objects.
        :return: distance.
        """
        return float(min(obj.polygon.distance(Point(*pose)) for obj in map_objects))

    def is_corner_far_from_drivable_area(
        self, map_api: AbstractMap, center_lane_lane_connector: List[GraphEdgeMapObject], ego_corner: Point2D
    ) -> bool:
        """
        Return a boolean that shows if ego_corner is far from drivable area according to the threshold.
        :param map_api: map api.
        :param center_lane_lane_connector: ego's center route obj in iteration.
        :param ego_corner: one of ego's corners.
        :return: boolean is_corner_far_from_drivable_area.
        """
        if center_lane_lane_connector:
            distance = self.compute_distance_to_map_objects_list(ego_corner, center_lane_lane_connector)
            if distance < self._max_violation_threshold:
                return False

        id_distance_tuple = map_api.get_distance_to_nearest_map_object(ego_corner, layer=SemanticMapLayer.DRIVABLE_AREA)

        return id_distance_tuple[1] is None or id_distance_tuple[1] >= self._max_violation_threshold

    def compute_violation_for_iteration(
        self,
        map_api: AbstractMap,
        ego_corners: List[Point2D],
        corners_lane_lane_connector: CornersGraphEdgeMapObject,
        center_lane_lane_connector: List[GraphEdgeMapObject],
        far_from_drivable_area: bool,
    ) -> Tuple[bool, bool]:
        """
        Compute violation of drivable area for an iteration.
        :param map_api: map api.
        :param ego_corners: 4 corners of ego (FL, RL, RR, FR) in iteration.
        :param corners_lane_lane_connector: object holding corners route objects.
        :param center_lane_lane_connector: ego's center route obj in iteration.
        :param far_from_drivable_area: boolean showing if ego got far from drivable_area in a previous iteration.
        :return: booleans not_in_drivable_area, far_from_drivable_area.
        """
        outside_drivable_area_objs = [
            ind
            for ind, obj in enumerate(corners_lane_lane_connector)
            if self.not_in_drivable_area_with_route_object(ego_corners[ind], obj, map_api)
        ]

        not_in_drivable_area = len(outside_drivable_area_objs) > 0

        far_from_drivable_area = far_from_drivable_area or any(
            self.is_corner_far_from_drivable_area(map_api, center_lane_lane_connector, ego_corners[ind])
            for ind in outside_drivable_area_objs
        )

        return (not_in_drivable_area, far_from_drivable_area)

    def extract_metric(self, history: SimulationHistory) -> Tuple[List[float], bool]:
        """
        Extract the drivable area violations from the history of Ego poses to evaluate drivable area compliance.
        :param history: SimulationHistory.
        :param corners_lane_lane_connector_list: List of corners lane and lane connectors.
        :return: list of float that shows if corners are in drivable area.
        """
        # Get ego states
        ego_states = history.extract_ego_state
        map_api = history.map_api
        all_ego_corners = extract_ego_corners(ego_states)  # 4 corners of oriented box (FL, RL, RR, FR)
        corners_lane_lane_connector_list = self._lane_change_metric.corners_route
        center_route = self._lane_change_metric.ego_driven_route

        corners_in_drivable_area = []
        far_from_drivable_area = False

        for ego_corners, corners_lane_lane_connector, center_lane_lane_connector in zip(
            all_ego_corners, corners_lane_lane_connector_list, center_route
        ):
            not_in_drivable_area, far_from_drivable_area = self.compute_violation_for_iteration(
                map_api, ego_corners, corners_lane_lane_connector, center_lane_lane_connector, far_from_drivable_area
            )
            corners_in_drivable_area.append(float(not not_in_drivable_area))

        return corners_in_drivable_area, far_from_drivable_area

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return: the estimated metric.
        """
        corners_in_drivable_area, far_from_drivable_area = self.extract_metric(history=history)

        statistics = [
            Statistic(
                name=f'{self.name}',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=float(not far_from_drivable_area),
                type=MetricStatisticsType.BOOLEAN,
            )
        ]
        self.results = self._construct_metric_results(
            metric_statistics=statistics, scenario=scenario, metric_score_unit=self._metric_score_unit
        )

        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_series = TimeSeries(unit='boolean', time_stamps=list(time_stamps), values=corners_in_drivable_area)
        corners_statistics = [
            Statistic(
                name='corners_in_drivable_area',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=float(np.all(corners_in_drivable_area)),
                type=MetricStatisticsType.BOOLEAN,
            )
        ]

        corners_statistics_result = MetricStatistics(
            metric_computator=self.name,
            name='corners_in_drivable_area',
            statistics=corners_statistics,
            time_series=time_series,
            metric_category=self.category,
        )
        # Save to load in high level metrics
        self.results.append(corners_statistics_result)

        return self.results
