import logging
from typing import List, Optional

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map_objects import GraphEdgeMapObject
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.route_extractor import get_distance_of_closest_baseline_point_to_its_start
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory

logger = logging.getLogger(__name__)


class DrivingDirectionComplianceStatistics(MetricBase):
    """Driving direction compliance metric.
    This metric traces if ego has been driving against the traffic flow more than some threshold during some time interval of ineterst.
    """

    def __init__(
        self,
        name: str,
        category: str,
        lane_change_metric: EgoLaneChangeStatistics,
        driving_direction_compliance_threshold: float = 2,
        driving_direction_violation_threshold: float = 6,
        time_horizon: float = 1,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initialize the DrivingDirectionComplianceStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param lane_change_metric: Lane change metric.
        :param driving_direction_compliance_threshold: Driving in opposite direction up to this threshold isn't considered violation
        :param driving_direction_violation_threshold: Driving in opposite direction above this threshold isn't tolerated
        :param time_horizon: Movement of the vehicle along baseline direction during a horizon time_horizon is
        considered for evaluation.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._lane_change_metric = lane_change_metric
        self._driving_direction_compliance_threshold = driving_direction_compliance_threshold
        self._driving_direction_violation_threshold = driving_direction_violation_threshold
        self._time_horizon = time_horizon

    @staticmethod
    def _extract_metric(
        ego_poses: List[Point2D], ego_driven_route: List[List[GraphEdgeMapObject]], n_horizon: int
    ) -> List[float]:
        """Compute the movement of ego during the past n_horizon samples along the direction of baselines.
        :param ego_poses: List of  ego poses.
        :param ego_driven_route: List of lanes/lane_connectors ego belongs to.
        :param n_horizon: Number of samples to sum the movement over.
        :return: A list of floats including ego's overall movements in the past n_horizon samples.
        """
        progress_along_baseline = []
        distance_to_start = None
        prev_distance_to_start = None
        prev_route_obj_id = None
        # If the first pose belongs to a lane/lane_connector store the id in prev_route_obj_id
        if ego_driven_route[0]:
            prev_route_obj_id = ego_driven_route[0][0].id

        # for each pose in the driven_trajectory compute the progress along the baseline of the corresponding lane/lane_connector in driven_route
        for ego_pose, ego_route_object in zip(ego_poses, ego_driven_route):
            # If pose isn't assigned a lane/lane_connector, there's no driving direction:
            if not ego_route_object:
                progress_along_baseline.append(0.0)
                continue
            # If the lane/lane_conn ego is in hasn't changed since last iteration compute the progress along its baseline
            # by subtracting its current distance to baseline's starting point from its distace in the previous iteration
            if prev_route_obj_id and ego_route_object[0].id == prev_route_obj_id:
                distance_to_start = get_distance_of_closest_baseline_point_to_its_start(
                    ego_route_object[0].baseline_path, ego_pose
                )
                # If prev_distance_to_start is set, compute the progress by subtracting distance_to_start from it, o.w set it to use in the next iteration
                progress_made = (
                    distance_to_start - prev_distance_to_start
                    if prev_distance_to_start is not None and distance_to_start
                    else 0.0
                )
                progress_along_baseline.append(progress_made)
                prev_distance_to_start = distance_to_start
            else:
                # Reset the parameters when ego first enters a lane/lane-connector
                distance_to_start = None
                prev_distance_to_start = None
                progress_along_baseline.append(0.0)
                prev_route_obj_id = ego_route_object[0].id

        # Compute progress over n_horizon last samples for each time point
        progress_over_n_horizon = [
            sum(progress_along_baseline[max(0, ind - n_horizon) : ind + 1])
            for ind, _ in enumerate(progress_along_baseline)
        ]
        return progress_over_n_horizon

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
        Return the driving direction compliance metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return: driving direction compliance statistics.
        """
        ego_states = history.extract_ego_state
        ego_poses = extract_ego_center(ego_states)
        ego_driven_route = self._lane_change_metric.ego_driven_route

        ego_timestamps = extract_ego_time_point(ego_states)
        n_horizon = int(self._time_horizon * 1e6 / np.mean(np.diff(ego_timestamps)))
        progress_over_interval = self._extract_metric(ego_poses, ego_driven_route, n_horizon)

        max_negative_progress_over_interval = abs(min(progress_over_interval))
        if max_negative_progress_over_interval < self._driving_direction_compliance_threshold:
            driving_direction_score = 1.0
        elif max_negative_progress_over_interval < self._driving_direction_violation_threshold:
            driving_direction_score = 0.5
        else:
            driving_direction_score = 0.0

        time_series = TimeSeries(
            unit="progress_along_driving_direction_in_last_" + f'{self._time_horizon}' + "_seconds_[m]",
            time_stamps=list(ego_timestamps),
            values=list(progress_over_interval),
        )

        statistics = [
            Statistic(
                name=f'{self.name}' + '_score',
                unit='value',
                value=float(driving_direction_score),
                type=MetricStatisticsType.VALUE,
            ),
            Statistic(
                name="min_progress_along_driving_direction_in_" + f'{self._time_horizon}' + "_second_interval",
                unit='meters',
                value=float(-max_negative_progress_over_interval),
                type=MetricStatisticsType.MIN,
            ),
        ]

        # Find results and save to re-use in high level metrics
        self.results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=statistics,
            scenario=scenario,
            time_series=time_series,
            metric_score_unit=self.metric_score_unit,
        )

        return self.results
