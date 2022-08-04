from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from shapely.geometry import CAP_STYLE, LineString, Polygon

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import Lane, StopLine
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.base.violation_metric_base import ViolationMetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from nuplan.planning.simulation.path.utils import trim_path_up_to_progress


@dataclass
class VelocityRecord:
    """
    Class to track velocity and distance to stop line at a timestamp.
    """

    velocity: float  # [m/s^2], Velocity at the current timestamp
    timestamp: int  # timestamp
    distance_to_stop_line: float  # [m], Distance to the stop line


@dataclass
class VelocityData:
    """
    Class to track VelocityRecord over the simulation history.
    """

    data: List[VelocityRecord]

    def add_data(self, velocity: float, timestamp: int, distance_to_stop_line: float) -> None:
        """
        Add new data to the list
        :param velocity: [m/s^2], Velocity at the current timestamp
        :param timestamp: Timestamp
        :param distance_to_stop_line: [m], Distance to the stop line.
        """
        if self.data is None:
            self.data = []

        self.data.append(
            VelocityRecord(velocity=velocity, timestamp=timestamp, distance_to_stop_line=distance_to_stop_line)
        )

    @property
    def velocity_np(self) -> npt.NDArray[np.float32]:
        """
        Velocity in numpy representation.
        """
        return np.asarray([data.velocity for data in self.data])

    @property
    def timestamp_np(self) -> npt.NDArray[np.int32]:
        """
        Timestamp in numpy representation.
        """
        return np.asarray([data.timestamp for data in self.data])

    @property
    def distance_to_stop_line_np(self) -> npt.NDArray[np.float32]:
        """
        Distance to stop line in numpy representation.
        """
        return np.asarray([data.distance_to_stop_line for data in self.data])

    @property
    def min_distance_stop_line_record(self) -> VelocityRecord:
        """
        Return velocity record of minimum distance stop line
        :return A velocity record.
        """
        distance_to_stop_line = self.distance_to_stop_line_np
        index = np.argmin(distance_to_stop_line)
        return self.data[int(index)]

    @property
    def min_velocity_record(self) -> VelocityRecord:
        """
        Return minimum velocity record
        :return A velocity record.
        """
        index = np.argmin(self.velocity_np)
        return self.data[int(index)]


class EgoStopAtStopLineStatistics(ViolationMetricBase):
    """
    Ego stopped at stop line metric.
    """

    def __init__(
        self,
        name: str,
        category: str,
        max_violation_threshold: int,
        distance_threshold: float,
        velocity_threshold: float,
    ) -> None:
        """
        Initializes the EgoProgressAlongExpertRouteStatistics class
        Rule formulation: 1. Get the nearest stop polygon (less than the distance threshold).
                          2. Check if the stop polygon is in any lanes.
                          3. Check if front corners of ego cross the stop polygon.
                          4. Check if no any leading agents.
                          5. Get min_velocity(distance_stop_line) until the ego leaves the stop polygon.
        :param name: Metric name
        :param category: Metric category
        :param max_violation_threshold: Maximum threshold for the violation when computing the score
        :param distance_threshold: Distances between ego front side and stop line lower than this threshold
        assumed to be the first vehicle before the stop line
        :param velocity_threshold: Velocity threshold to consider an ego stopped.
        """
        super().__init__(name=name, category=category, max_violation_threshold=max_violation_threshold)
        self._distance_threshold = distance_threshold
        self._velocity_threshold = velocity_threshold
        self._stopping_velocity_data: List[VelocityData] = []
        self._previous_stop_polygon_fid: Optional[str] = None

    @staticmethod
    def get_nearest_stop_line(map_api: AbstractMap, ego_pose_front: LineString) -> Optional[Tuple[str, Polygon]]:
        """
        Retrieve the nearest stop polygon
        :param map_api: AbstractMap map api
        :param ego_pose_front: Ego pose front corner line
        :return Nearest stop polygon fid if distance is less than the threshold.
        """
        center_x, center_y = ego_pose_front.centroid.xy
        center = Point2D(center_x[0], center_y[0])

        # If ego is not in lane, then we do not need to proceed.
        if not map_api.is_in_layer(center, layer=SemanticMapLayer.LANE):
            return None

        stop_line_fid, distance = map_api.get_distance_to_nearest_map_object(center, SemanticMapLayer.STOP_LINE)

        if stop_line_fid is None:
            return None

        stop_line: StopLine = map_api.get_map_object(stop_line_fid, SemanticMapLayer.STOP_LINE)
        lane: Optional[Lane] = map_api.get_one_map_object(center, SemanticMapLayer.LANE)

        # Check if the stop polygon intersects with the ego lane
        if lane is not None:
            return stop_line_fid, stop_line.polygon if stop_line.polygon.intersects(lane.polygon) else None

        return None

    @staticmethod
    def check_for_leading_agents(detections: Observation, ego_state: EgoState, map_api: AbstractMap) -> bool:
        """
        Get the nearest leading agent
        :param detections: Detection class
        :param ego_state: Ego in oriented box representation
        :param map_api: AbstractMap api
        :return True if there is a leading agent, False otherwise
        """
        if isinstance(detections, DetectionsTracks):

            if len(detections.tracked_objects.tracked_objects) == 0:
                return False

            ego_agent = ego_state.agent

            # Check if any missing instance token
            for index, box in enumerate(detections.tracked_objects):
                if box.token is None:
                    box.token = str(index + 1)
            scene_objects: List[SceneObject] = [ego_agent]
            scene_objects.extend([scene_object for scene_object in detections.tracked_objects])

            occupancy_map = STRTreeOccupancyMapFactory.get_from_boxes(scene_objects)
            agent_states = {
                scene_object.token: StateSE2(
                    x=scene_object.center.x, y=scene_object.center.y, heading=scene_object.center.heading
                )
                for scene_object in scene_objects
            }
            ego_pose: StateSE2 = agent_states['ego']
            lane = map_api.get_one_map_object(ego_pose, SemanticMapLayer.LANE)

            # Construct ego's path to go
            ego_baseline = lane.baseline_path
            ego_progress = ego_baseline.get_nearest_arc_length_from_position(ego_pose)
            progress_path = create_path_from_se2(ego_baseline.discrete_path)
            ego_path_to_go = trim_path_up_to_progress(progress_path, ego_progress)

            # Check for intersection between the path and the any other agents
            ego_path_to_go = path_to_linestring(ego_path_to_go)
            intersecting_agents = occupancy_map.intersects(
                ego_path_to_go.buffer((scene_objects[0].box.width / 2), cap_style=CAP_STYLE.flat)
            )

            # If there are other agents ahead of the ego in the same lane there should be at least two agents
            if intersecting_agents.size > 1:
                return True

        return False

    def _compute_velocity_statistics(self, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Compute statistics in each stop line
        :param scenario: Scenario running this metric
        :return A list of metric statistics.
        """
        if not self._stopping_velocity_data:
            return []

        mean_ego_min_distance_to_stop_line = []
        mean_ego_min_velocity_before_stop_line = []
        aggregated_timestamp_velocity = []
        aggregated_timestamps = []
        ego_stop_status = []

        for velocity_data in self._stopping_velocity_data:

            # Stop at stop line success rate, we check the velocity of minimum distance to the stop line in each record
            min_distance_velocity_record = velocity_data.min_distance_stop_line_record
            mean_ego_min_distance_to_stop_line.append(min_distance_velocity_record.distance_to_stop_line)
            mean_ego_min_velocity_before_stop_line.append(min_distance_velocity_record.velocity)

            if (
                min_distance_velocity_record.distance_to_stop_line < self._distance_threshold
                and min_distance_velocity_record.velocity < self._velocity_threshold
            ):
                stop_status = True
            else:
                stop_status = False

            ego_stop_status.append(stop_status)
            aggregated_timestamp_velocity.append(velocity_data.velocity_np)
            aggregated_timestamps.append(velocity_data.timestamp_np)

        # Aggregate
        statistics = [
            Statistic(
                name='number_of_ego_stop_before_stop_line',
                unit=MetricStatisticsType.COUNT.unit,
                value=sum(ego_stop_status),
                type=MetricStatisticsType.COUNT,
            ),
            Statistic(
                name='number_of_ego_before_stop_line',
                unit=MetricStatisticsType.COUNT.unit,
                value=len(ego_stop_status),
                type=MetricStatisticsType.COUNT,
            ),
            Statistic(
                name='mean_ego_min_distance_to_stop_line',
                unit='meters',
                value=float(np.mean(mean_ego_min_distance_to_stop_line)),
                type=MetricStatisticsType.VALUE,
            ),
            Statistic(
                name='mean_ego_min_velocity_before_stop_line',
                unit='meters_per_second_squared',
                value=float(np.mean(mean_ego_min_velocity_before_stop_line)),
                type=MetricStatisticsType.VALUE,
            ),
        ]

        aggregated_timestamp_velocity = np.hstack(aggregated_timestamp_velocity)  # type: ignore
        aggregated_timestamps = np.hstack(aggregated_timestamps)  # type: ignore
        velocity_time_series = TimeSeries(
            unit='meters_per_second_squared',
            time_stamps=list(aggregated_timestamps),
            values=list(aggregated_timestamp_velocity),
        )

        results = self._construct_metric_results(
            metric_statistics=statistics, time_series=velocity_time_series, scenario=scenario
        )
        return results  # type: ignore

    def _save_stopping_velocity(
        self,
        current_stop_polygon_fid: str,
        history_data: SimulationHistorySample,
        stop_polygon_in_lane: Polygon,
        ego_pose_front: LineString,
    ) -> None:
        """
        Save velocity, timestamp and distance to a stop line if the ego is stopping
        :param current_stop_polygon_fid: Current stop polygon fid
        :param history_data: History sample data at current timestamp
        :param stop_polygon_in_lane: The stop polygon where the ego is in
        :param ego_pose_front: Front line string (front right corner and left corner) of the ego.
        """
        # Distance between the front stop line and ego's front footprint
        stop_line: LineString = LineString(stop_polygon_in_lane.exterior.coords[:2])
        distance_ego_front_stop_line = stop_line.distance(ego_pose_front)
        current_velocity = history_data.ego_state.dynamic_car_state.speed
        current_timestamp = history_data.ego_state.time_point.time_us

        # If it is in the same stop polygon
        if current_stop_polygon_fid == self._previous_stop_polygon_fid:
            self._stopping_velocity_data[-1].add_data(
                velocity=current_velocity,
                timestamp=current_timestamp,
                distance_to_stop_line=distance_ego_front_stop_line,
            )
        else:
            # If it is a new stop polygon
            self._previous_stop_polygon_fid = current_stop_polygon_fid
            velocity_data = VelocityData([])
            velocity_data.add_data(
                velocity=current_velocity,
                timestamp=current_timestamp,
                distance_to_stop_line=distance_ego_front_stop_line,
            )
            self._stopping_velocity_data.append(velocity_data)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the ego stopped at stop line metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated ego stopped at stop line metric.
        """
        # Extract ego pose in box3d.
        ego_states: List[EgoState] = history.extract_ego_state

        # Get egos' front footprint
        ego_pose_fronts: List[LineString] = [
            LineString(
                [
                    state.car_footprint.oriented_box.geometry.exterior.coords[0],
                    state.car_footprint.oriented_box.geometry.exterior.coords[3],
                ]
            )
            for state in ego_states
        ]

        # Get stop polygons
        scenario_map: AbstractMap = history.map_api

        # Get the nearest stop polygon.
        for ego_pose_front, ego_state, history_data in zip(ego_pose_fronts, ego_states, history.data):

            # 1) Get the nearest stop polygon if it is in the same lane as ego.
            stop_polygon_info: Optional[Tuple[str, Polygon]] = self.get_nearest_stop_line(
                map_api=scenario_map, ego_pose_front=ego_pose_front
            )

            if stop_polygon_info is None:
                continue

            fid, stop_polygon_in_lane = stop_polygon_info

            # 2) Check if front corners cross the stop polygon
            ego_pose_front_stop_polygon_distance: float = ego_pose_front.distance(stop_polygon_in_lane)

            # The front corners cross the stop polygon if distance is zero
            if ego_pose_front_stop_polygon_distance != 0:
                continue

            # 3) Check if no any leading agents.
            detections: Observation = history_data.observation
            has_leading_agent = self.check_for_leading_agents(
                detections=detections, ego_state=ego_state, map_api=scenario_map
            )

            # Skip if there is any leading agent in the stop polygon (means the ego is not the first vehicle)
            if has_leading_agent:
                continue

            # 4) Accumulate velocity
            self._save_stopping_velocity(
                current_stop_polygon_fid=fid,
                history_data=history_data,
                stop_polygon_in_lane=stop_polygon_in_lane,
                ego_pose_front=ego_pose_front,
            )

        results = self._compute_velocity_statistics(scenario=scenario)
        return results
