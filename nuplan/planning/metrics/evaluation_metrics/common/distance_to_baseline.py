from typing import List

import numpy as np
from shapely.geometry import Point

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, TimeSeries
from nuplan.planning.metrics.utils.route_extractor import get_route
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class DistanceToBaselineStatistics(MetricBase):
    """Statistics on distance of center of ego from nearest baseline."""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializes the DistanceToBaselineStatistics class
        :param name: Metric name
        :param category: Metric category.
        """
        super().__init__(name=name, category=category)

    @staticmethod
    def compute_distance_to_route_baseline(map_api: AbstractMap, poses: List[Point2D]) -> List[float]:
        """
        Returns minimum distances of each ego pose to the baseline of a lane or lane_connector that it
        belongs to one, if it does not belong to any lane or lane_connector inf is returned
        :param map_api: a map
        :param ego_poses: list of ego poses
        :return list of ditances to baseline, or inf.
        """
        # Get the list of lane or lane_connectors ego belongs to.
        ego_route = get_route(map_api=map_api, poses=poses)

        # For each (route_obj, pose), if route_obj is not None, compute the distance of pose from its
        # baseline, otherwise set distance to inf
        distances = []
        for route_obj, pose in zip(ego_route, poses):
            if len(route_obj) == 0:
                distances.append(np.inf)
                continue
            baseline_paths = [one_route_obj.baseline_path() for one_route_obj in route_obj]
            dist_to_route = min(
                baseline_path.linestring.distance(Point(pose.x, pose.y)) for baseline_path in baseline_paths
            )
            distances.append(dist_to_route)
        return distances

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        # Extract xy coordinates of center of ego from history.
        ego_states = history.extract_ego_state
        ego_poses = extract_ego_center(ego_states)

        # Compute distance of center poses from the baseline of route objects.
        distance_to_baseline = self.compute_distance_to_route_baseline(map_api=history.map_api, poses=ego_poses)

        ego_timestamps = extract_ego_time_point(ego_states)

        time_series = TimeSeries(unit='meters', time_stamps=list(ego_timestamps), values=list(distance_to_baseline))
        statistics_type_list = [MetricStatisticsType.MAX, MetricStatisticsType.MEAN]

        metric_statistics = self._compute_time_series_statistic(
            time_series=time_series, statistics_type_list=statistics_type_list
        )

        results = self._construct_metric_results(
            metric_statistics=metric_statistics, scenario=scenario, time_series=time_series
        )
        return results  # type: ignore
