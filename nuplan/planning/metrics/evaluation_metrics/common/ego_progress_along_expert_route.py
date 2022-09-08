import logging
from typing import List, Optional

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map_objects import PolylineMapObject
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.route_extractor import (
    RouteBaselineRoadBlockPair,
    RouteRoadBlockLinkedList,
    get_distance_of_closest_baseline_point_to_its_start,
    get_route,
    get_route_baseline_roadblock_linkedlist,
    get_route_simplified,
)
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory

logger = logging.getLogger(__name__)


class PerFrameProgressAlongRouteComputer:
    """Class that computes progress per frame along a route."""

    def __init__(self, route_roadblocks: RouteRoadBlockLinkedList):
        """Class initializer
        :param route_roadblocks: A route roadblock linked list.
        """
        self.curr_roadblock_pair = route_roadblocks.head
        self.progress = [float(0)]
        self.prev_distance_to_start = float(0)
        self.next_roadblock_pair: Optional[RouteBaselineRoadBlockPair] = None
        self.skipped_roadblock_pair: Optional[RouteBaselineRoadBlockPair] = None

    @staticmethod
    def get_some_baseline_point(baseline: PolylineMapObject, ind: str) -> Optional[Point2D]:
        """Gets the first or last point on a given baselinePath
        :param baseline: A baseline path
        :param ind: Either 'last' or 'first' strings to show which point function should return
        :return: A point.
        """
        if ind == 'last':
            return Point2D(baseline.linestring.xy[0][-1], baseline.linestring.xy[1][-1])
        elif ind == 'first':
            return Point2D(baseline.linestring.xy[0][0], baseline.linestring.xy[1][0])
        else:
            raise ValueError('invalid position argument')

    def compute_progress_for_skipped_road_block(self) -> float:
        """Computes progress for skipped road_blocks (when ego pose exits one road block in a route and it does not
        enter the next one)
        :return: progress_for_skipped_roadblock
        """
        assert self.next_roadblock_pair is not None
        if self.skipped_roadblock_pair:
            prev_roadblock_last_point = self.get_some_baseline_point(self.skipped_roadblock_pair.base_line, 'last')
        else:
            prev_roadblock_last_point = self.get_some_baseline_point(self.curr_roadblock_pair.base_line, 'last')

        self.skipped_roadblock_pair = self.next_roadblock_pair
        skipped_distance_to_start = get_distance_of_closest_baseline_point_to_its_start(
            self.skipped_roadblock_pair.base_line, prev_roadblock_last_point
        )
        self.next_roadblock_pair = self.next_roadblock_pair.next
        next_roadblock_first_point = self.get_some_baseline_point(self.next_roadblock_pair.base_line, 'first')
        next_baseline_start_dist_to_skipped = get_distance_of_closest_baseline_point_to_its_start(
            self.skipped_roadblock_pair.base_line, next_roadblock_first_point
        )

        progress_for_skipped_roadblock: float = next_baseline_start_dist_to_skipped - skipped_distance_to_start
        return progress_for_skipped_roadblock

    def get_progress_including_skipped_roadblocks(
        self, ego_pose: Point2D, progress_for_skipped_roadblock: float
    ) -> float:
        """Computes ego's progress when it first enters a new road-block in the route by considering possible progress
        for roadblocks it has skipped as multi_block_progress = (progress along the baseline of prev ego roadblock)
        + (progress along the baseline of the roadblock ego is in now) + (progress along skipped roadblocks if any).
        :param ego_pose: ego pose
        :param progress_for_skipped_roadblock: Prgoress for skipped road_blocks (zero if no roadblocks is skipped)
        :return: multi_block_progress
        """
        assert self.next_roadblock_pair is not None
        # progress in previous roadblock compared to previous pose in that roadblock is equal
        # to length of prev_baseline - distance of prev pose to start of baseline
        progress_in_prev_roadblock = self.curr_roadblock_pair.base_line.linestring.length - self.prev_distance_to_start
        prev_roadblock_last_point = self.get_some_baseline_point(self.curr_roadblock_pair.base_line, 'last')

        self.curr_roadblock_pair = self.next_roadblock_pair
        # distance to start of the baseline corresponding to roadblock ego_pose is in
        distance_to_start = get_distance_of_closest_baseline_point_to_its_start(
            self.curr_roadblock_pair.base_line, ego_pose
        )
        # distance of last_baseline_point to start of the baseline corresponding to roadblock ego_pose is in
        last_baseline_point_dist_to_start = get_distance_of_closest_baseline_point_to_its_start(
            self.curr_roadblock_pair.base_line, prev_roadblock_last_point
        )
        # progress in new roadblock is computed as the difference between previous two variables
        progress_in_new_roadblock = distance_to_start - last_baseline_point_dist_to_start
        # Total progress is the progress in previous and new roadblocks and accounts for skipped roadblocks
        multi_block_progress = progress_in_prev_roadblock + progress_in_new_roadblock + progress_for_skipped_roadblock
        self.prev_distance_to_start = distance_to_start
        return float(multi_block_progress)

    def get_multi_block_progress(self, ego_pose: Point2D) -> float:
        """When ego pose exits previous roadblock this function takes next road blocks in the expert route one by one
        until it finds one (if any) that pose belongs to. Once found, ego progress for multiple roadblocks including
        possible skipped roadblocks is computed and returned
        :param ego_pose: ego pose
        :return: multi block progress
        """
        multi_block_progress = float(0)
        progress_for_skipped_roadblocks = float(0)
        self.next_roadblock_pair = self.curr_roadblock_pair.next
        self.skipped_roadblock_pair = None

        while self.next_roadblock_pair is not None:
            if self.next_roadblock_pair.road_block.contains_point(ego_pose):
                multi_block_progress = self.get_progress_including_skipped_roadblocks(
                    ego_pose, progress_for_skipped_roadblocks
                )
                break
            elif not self.next_roadblock_pair.next:
                break
            else:
                progress_for_skipped_roadblocks += self.compute_progress_for_skipped_road_block()
        return multi_block_progress

    def __call__(self, ego_poses: List[Point2D]) -> List[float]:
        """
        Computes per frame progress along the route baselines for ego poses
        :param ego_poses: ego poses
        :return: progress along the route.
        """
        # Compute distance to the beginning of the baseline that corresponds to ego's initial pose
        self.prev_distance_to_start = get_distance_of_closest_baseline_point_to_its_start(
            self.curr_roadblock_pair.base_line, ego_poses[0]
        )

        # For each pose if it's in the last taken road_block compute the new distance to the
        # beginning of the baseline and take progress as the difference with the previous distance.
        # O.w, take the next road blocks in the expert route to find one the pose belongs to and take
        # progress as the multi_block_progress = (progress along the baseline of prev ego roadblock) +
        # (progress along the baseline of the roadblock ego is in now) + (progress along skipped roadblocks if any).
        for ego_pose in ego_poses[1:]:
            if self.curr_roadblock_pair.road_block.contains_point(ego_pose):
                distance_to_start = get_distance_of_closest_baseline_point_to_its_start(
                    self.curr_roadblock_pair.base_line, ego_pose
                )
                self.progress.append(distance_to_start - self.prev_distance_to_start)
                self.prev_distance_to_start = distance_to_start
            else:
                multi_block_progress = self.get_multi_block_progress(ego_pose)
                self.progress.append(multi_block_progress)

        return self.progress


class EgoProgressAlongExpertRouteStatistics(MetricBase):
    """Ego progress along the expert route metric."""

    def __init__(
        self, name: str, category: str, score_progress_threshold: float = 2, metric_score_unit: Optional[str] = None
    ) -> None:
        """
        Initializes the EgoProgressAlongExpertRouteStatistics class
        :param name: Metric name
        :param category: Metric category
        :param score_progress_threshold: Progress distance threshold for the score.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._score_progress_threshold = score_progress_threshold

        # Store results to re-use in high level metrics
        self.results: List[MetricStatistics] = []

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[-1].value)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the ego progress along the expert route metric
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric
        :return: Ego progress along expert route statistics.
        """
        ego_states = history.extract_ego_state
        ego_poses = extract_ego_center(ego_states)

        expert_states = scenario.get_expert_ego_trajectory()
        expert_poses = extract_ego_center(expert_states)

        # Get expert's route and simplify it by removing repeated consequtive route objects
        expert_route = get_route(map_api=history.map_api, poses=expert_poses)
        expert_route_simplified = get_route_simplified(expert_route)
        if not expert_route_simplified:
            # If expert_route_simplified is empty (no lanes/lane_connectors could be assigned to expert, e.g. when expert is in car_park),
            # and so no route info is available, planner gets the score for this metric:
            statistics = [
                Statistic(
                    name='expert_total_progress_along_route',
                    unit='meters',
                    value=0.0,
                    type=MetricStatisticsType.VALUE,
                ),
                Statistic(
                    name='ego_expert_progress_along_route_ratio',
                    unit=MetricStatisticsType.RATIO.unit,
                    value=1.0,
                    type=MetricStatisticsType.RATIO,
                ),
            ]
            # Find results and save to re-use in high level metrics
            self.results = self._construct_metric_results(metric_statistics=statistics, scenario=scenario)
        else:
            # Find route's baselines and roadblocks and generate a linked list of baseline-roadblock pairs
            route_baseline_roadblock_pairs = get_route_baseline_roadblock_linkedlist(
                history.map_api, expert_route_simplified
            )

            # Compute ego's progress along the route
            ego_progress_computer = PerFrameProgressAlongRouteComputer(route_roadblocks=route_baseline_roadblock_pairs)
            ego_progress = ego_progress_computer(ego_poses=ego_poses)
            overall_ego_progress = np.sum(ego_progress)

            # Compute expert's progress as baseline for comparison.
            expert_progress_computer = PerFrameProgressAlongRouteComputer(
                route_roadblocks=route_baseline_roadblock_pairs
            )
            expert_progress = expert_progress_computer(ego_poses=expert_poses)
            overall_expert_progress = np.sum(expert_progress)

            # Ego is not allowed to fully drive backwards in our scenarios. Due to noise in data, in some scenarios where ego is stopped during the scenario
            # we may get a small negative value in overall_ego_progress up to -self._score_progress_threshold. We set ego's to expert progress ratio to 0 if
            # overall_ego_progress is less than this negative progress threshold
            if overall_ego_progress < -self._score_progress_threshold:
                ego_expert_progress_along_route_ratio = 0

            else:
                # Find the ratio of ego's to expert progress along the route and saturate it in [0,1]. We set this ratio to 1 if expert does not move
                # more than some minimum progress threshold (e.g. expert is stopped for a red light for the entire scenario duration and so is the proposed ego trajectory)
                ego_expert_progress_along_route_ratio = min(
                    1.0,
                    max(overall_ego_progress, self._score_progress_threshold)
                    / max(overall_expert_progress, self._score_progress_threshold),
                )

            ego_timestamps = extract_ego_time_point(ego_states)

            time_series = TimeSeries(unit='meters', time_stamps=list(ego_timestamps), values=list(ego_progress))
            statistics = [
                Statistic(
                    name='expert_total_progress_along_route',
                    unit='meters',
                    value=float(overall_expert_progress),
                    type=MetricStatisticsType.VALUE,
                ),
                Statistic(
                    name='ego_total_progress_along_route',
                    unit='meters',
                    value=float(overall_ego_progress),
                    type=MetricStatisticsType.VALUE,
                ),
                Statistic(
                    name='ego_expert_progress_along_route_ratio',
                    unit=MetricStatisticsType.RATIO.unit,
                    value=ego_expert_progress_along_route_ratio,
                    type=MetricStatisticsType.RATIO,
                ),
            ]
            # Find results and save to re-use in high level metrics
            self.results = self._construct_metric_results(
                metric_statistics=statistics,
                scenario=scenario,
                time_series=time_series,
                metric_score_unit=self.metric_score_unit,
            )

        return self.results
