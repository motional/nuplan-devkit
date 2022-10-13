import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import numpy.typing as npt

from nuplan.common.maps.abstract_map_objects import GraphEdgeMapObject
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic
from nuplan.planning.metrics.utils.route_extractor import (
    CornersGraphEdgeMapObject,
    extract_corners_route,
    get_common_or_connected_route_objs_of_corners,
    get_outgoing_edges_obj_dict,
    get_route,
    get_timestamps_in_common_or_connected_route_objs,
)
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory

logger = logging.getLogger(__name__)


@dataclass
class LaneChangeStartRecord:
    """
    Class used to keep track of the timestamp of beginning of a lane change and initial lane or
    initial lane connector(s) if lane change starts in areas annotated as intersection.
    """

    start_timestamp: int
    initial_lane: Optional[Set[GraphEdgeMapObject]]


@dataclass
class LaneChangeData:
    """
    Class used to store lane change data, contains the data on beginning of a lane change, its duration
    in micro seconds, target lane and whether it was successful or not.

    A lane change starts if at the previous timestamps ego was fully within a lane/lane connector (initial lane)
    and at the current timestamp either of the front corners enter another lane different than the initial lane.

    A lane change is complete if all corners of ego enter the same lane/lane connectors.

    A lane change is sucessful if all corners of ego enter a target lane that is different than the initial lane
    and fails if either of the following cases happens:
    1. Ego fully returns to the initial lane
    2. Ego enters nondrivable area before completing the lane change
    3. Scenario ends before completing the lane change.
    """

    start_data: LaneChangeStartRecord
    duration_us: float
    final_lane: Optional[Set[GraphEdgeMapObject]]
    success: bool


def _ego_starts_lane_change(
    initial_lane: Optional[Set[GraphEdgeMapObject]], start_timestamp: int
) -> Optional[LaneChangeStartRecord]:
    """
    Opens lane change window and stores the information
    :param initial_lane: Set of common/connected route objects of corners of ego at previous timestamp
    :param start_timestamp: The current timestamp
    :return information on starts of a lane change if exists, otherwise None.
    """
    # We don't consider lane change if in the previous timestamp corners were in different lane/lane connectors
    # or all corners were in nondrivable area
    return LaneChangeStartRecord(start_timestamp, initial_lane) if initial_lane else None


def _ego_ends_lane_change(
    open_lane_change: LaneChangeStartRecord, final_lane: Set[GraphEdgeMapObject], end_timestamp: int
) -> LaneChangeData:
    """
    Stores the information if ego ends a lane change
    :param open_lane_change: Record of the currently open lane change
    :param final_lane: Set of common/connected route objects of corners of ego when completing a lane change
    :param end_timestamp: The current timestamp
    :return LaneChangeData.
    """
    # Fail if ego exits the drivable area before completing the lane change, set final_lane as None
    if not final_lane:
        return LaneChangeData(
            open_lane_change, end_timestamp - open_lane_change.start_timestamp, final_lane=None, success=False
        )

    initial_lane = open_lane_change.initial_lane
    initial_lane_ids = {obj.id for obj in initial_lane}  # type: ignore
    initial_lane_out_edge_ids = set(get_outgoing_edges_obj_dict(initial_lane).keys())
    initial_lane_or_out_edge_ids = initial_lane_ids.union(initial_lane_out_edge_ids)
    final_lane_ids = {obj.id for obj in final_lane}

    return LaneChangeData(
        open_lane_change,
        end_timestamp - open_lane_change.start_timestamp,
        final_lane,
        success=False if len(set.intersection(initial_lane_or_out_edge_ids, final_lane_ids)) else True,
    )


def find_lane_changes(
    ego_timestamps: npt.NDArray[np.int32], common_or_connected_route_objs: List[Optional[Set[GraphEdgeMapObject]]]
) -> List[LaneChangeData]:
    """
    Extracts the lane changes in the scenario
    :param ego_timestamps: Array of times in time_us
    :param common_or_connected_route_objs: list of common or connected lane/lane connectors of corners
    :return List of lane change data in the scenario.
    """
    lane_changes: List[LaneChangeData] = []
    open_lane_change = None

    if common_or_connected_route_objs[0] is None:
        logging.debug("Scenario starts with corners in different route objects")

    for prev_ind, curr_obj in enumerate(common_or_connected_route_objs[1:]):
        # check there is no open lane change window
        if open_lane_change is None:
            # Check if current common obj is None (so corners are in different ojects)
            if curr_obj is None:
                open_lane_change = _ego_starts_lane_change(
                    initial_lane=common_or_connected_route_objs[prev_ind], start_timestamp=ego_timestamps[prev_ind + 1]
                )

        else:
            # Check if an open lane change ends and store the data
            if curr_obj is not None:
                lane_change_data = _ego_ends_lane_change(
                    open_lane_change, final_lane=curr_obj, end_timestamp=ego_timestamps[prev_ind + 1]
                )
                lane_changes.append(lane_change_data)
                open_lane_change = None

    # Fail lane change and close interval if the open lane change has not completed during the scenario, set
    # final_lane as None
    if open_lane_change:
        lane_changes.append(
            LaneChangeData(
                open_lane_change, ego_timestamps[-1] - open_lane_change.start_timestamp, final_lane=None, success=False
            )
        )

    return lane_changes


class EgoLaneChangeStatistics(MetricBase):
    """Statistics on lane change."""

    def __init__(self, name: str, category: str, max_fail_rate: float) -> None:
        """
        Initializes the EgoLaneChangeStatistics class
        :param name: Metric name
        :param category: Metric category
        :param max_fail_rate: maximum acceptable ratio of failed to total number of lane changes.
        """
        super().__init__(name=name, category=category)
        self._max_fail_rate = max_fail_rate

        # Store to re-use in high-level metrics
        self.ego_driven_route: List[List[Optional[GraphEdgeMapObject]]] = []
        self.corners_route: List[CornersGraphEdgeMapObject] = [CornersGraphEdgeMapObject([], [], [], [])]
        self.timestamps_in_common_or_connected_route_objs: List[int] = []
        self.results: List[MetricStatistics] = []

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the lane chane metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated lane change duration in micro seconds and status.
        """
        # Extract xy coordinates of center of ego from history.
        ego_states = history.extract_ego_state
        ego_poses = extract_ego_center(ego_states)

        # Get the list of lane or lane_connectors associated to ego at each time instance, and store to use in other metrics
        self.ego_driven_route = get_route(history.map_api, ego_poses)

        # Extract ego timepoints
        ego_timestamps = extract_ego_time_point(ego_states)

        # Extract corners of ego's footprint
        ego_footprint_list = [ego_state.car_footprint for ego_state in ego_states]

        # Extract corner lanes/lane connectors
        corners_route = extract_corners_route(history.map_api, ego_footprint_list)
        # Store to load in high level metrics
        self.corners_route = corners_route

        common_or_connected_route_objs = get_common_or_connected_route_objs_of_corners(corners_route)

        # Extract ego timepoints where its corners are in common_or_connected_route objs
        timestamps_in_common_or_connected_route_objs = get_timestamps_in_common_or_connected_route_objs(
            common_or_connected_route_objs, ego_timestamps
        )

        # Store to load in high level metrics
        self.timestamps_in_common_or_connected_route_objs = timestamps_in_common_or_connected_route_objs

        # Extract lane changes in the history
        lane_changes = find_lane_changes(ego_timestamps, common_or_connected_route_objs)

        if len(lane_changes) == 0:
            metric_statistics = [
                Statistic(
                    name=f"number_of_{self.name}",
                    unit=MetricStatisticsType.COUNT.unit,
                    value=0,
                    type=MetricStatisticsType.COUNT,
                ),
                Statistic(
                    name=f"{self.name}_fail_rate_below_threshold",
                    unit=MetricStatisticsType.BOOLEAN.unit,
                    value=True,
                    type=MetricStatisticsType.BOOLEAN,
                ),
            ]

        else:
            # Find lane change durations in seconds and ratio of number of failed to total number of lane changes
            lane_change_durations = [lane_change.duration_us * 1e-6 for lane_change in lane_changes]
            failed_lane_changes = [lane_change for lane_change in lane_changes if not lane_change.success]
            failed_ratio = len(failed_lane_changes) / len(lane_changes)
            fail_rate_below_threshold = 1 if self._max_fail_rate >= failed_ratio else 0
            metric_statistics = [
                Statistic(
                    name=f"number_of_{self.name}",
                    unit=MetricStatisticsType.COUNT.unit,
                    value=len(lane_changes),
                    type=MetricStatisticsType.COUNT,
                ),
                Statistic(
                    name=f"max_{self.name}_duration",
                    unit="seconds",
                    value=np.max(lane_change_durations),
                    type=MetricStatisticsType.MAX,
                ),
                Statistic(
                    name=f"avg_{self.name}_duration",
                    unit="seconds",
                    value=float(np.mean(lane_change_durations)),
                    type=MetricStatisticsType.MEAN,
                ),
                Statistic(
                    name=f"ratio_of_failed_{self.name}",
                    unit=MetricStatisticsType.RATIO.unit,
                    value=failed_ratio,
                    type=MetricStatisticsType.RATIO,
                ),
                Statistic(
                    name=f"{self.name}_fail_rate_below_threshold",
                    unit=MetricStatisticsType.BOOLEAN.unit,
                    value=bool(fail_rate_below_threshold),
                    type=MetricStatisticsType.BOOLEAN,
                ),
            ]

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=metric_statistics, time_series=None, scenario=scenario
        )

        self.results = results

        return results
