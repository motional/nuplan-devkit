from typing import Dict, List

import numpy as np

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.route_extractor import CornersGraphEdgeMapObject
from nuplan.planning.metrics.utils.state_extractors import extract_ego_corners, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class DrivableAreaViolationStatistics(MetricBase):
    """Statistics on drivable area violations of ego."""

    def __init__(self, name: str, category: str, lane_change_metric: EgoLaneChangeStatistics) -> None:
        """
        Initializes the DrivableAreaViolationStatistics class
        :param name: Metric name
        :param category: Metric category
        :param lane_change_metric: lane change metric
        """
        super().__init__(name=name, category=category)

        # Save to load in high level metrics
        self.results: List[MetricStatistics] = []
        self._lane_change_metric = lane_change_metric

    def extract_metric(
        self, history: SimulationHistory, corners_lane_laneconnector_list: List[CornersGraphEdgeMapObject]
    ) -> List[float]:
        """
        Extracts the drivable area violations from the history of Ego poses.
        :param history: SimulationHistory
        :param corners_lane_laneconnector_list: List of corners lane and lane connectors
        :return : list of float that shows if corners are in drivable area
        """
        # Get ego states
        ego_states = history.extract_ego_state
        all_ego_corners = extract_ego_corners(ego_states)  # 4 corners of oriented box (FL, RL, RR, FR)

        corners_not_in_drivable_area = []

        for ego_corners, corners_lane_laneconnector in zip(all_ego_corners, corners_lane_laneconnector_list):
            for corner_ind, corner_map_object in enumerate(corners_lane_laneconnector.__iter__()):
                not_in_drivable_area = False
                if not corner_map_object and not history.map_api.is_in_layer(
                    ego_corners[corner_ind], layer=SemanticMapLayer.DRIVABLE_AREA
                ):
                    not_in_drivable_area = True
                    break
            corners_not_in_drivable_area.append(float(not_in_drivable_area))

        return corners_not_in_drivable_area

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: TimeSeries,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[MetricStatisticsType.BOOLEAN].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return: the estimated metric.
        """
        # There are some corner cases in which pose can belong to nuplan lane_connectors but not be in drivable area, because
        # lane_connectors right now are inferred from baselines using some
        # Also we only check if corners are in drivable area, not any point on AV polygon.
        corners_route = self._lane_change_metric.corners_route
        corners_not_in_drivable_area = self.extract_metric(
            history=history, corners_lane_laneconnector_list=corners_route
        )

        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_series = TimeSeries(unit='boolean', time_stamps=list(time_stamps), values=corners_not_in_drivable_area)
        statistics = {
            MetricStatisticsType.BOOLEAN: Statistic(
                name=f'no_{self.name}',
                unit='boolean',
                value=float(not np.any(corners_not_in_drivable_area)),
            ),
            MetricStatisticsType.RATIO: Statistic(
                name=f'ratio_of_{self.name}_duration_to_scenario_duration',
                unit='ratio',
                value=sum(corners_not_in_drivable_area) / max(len(corners_not_in_drivable_area), 1),
            ),
        }

        results = self._construct_metric_results(
            metric_statistics=statistics, scenario=scenario, time_series=time_series
        )

        # Save to load in high level metrics
        self.results = results
        return results  # type: ignore
