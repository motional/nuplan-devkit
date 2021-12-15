from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import Lane, StopLine
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.geometry import quaternion_yaw
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections, Observation
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import create_path_from_se2, \
    ego_state_to_box_3d, path_to_linestring
from nuplan.planning.simulation.observation.smart_agents.occupancy_map.strtree_occupancy_map import \
    STRTreeOccupancyMapFactory
from nuplan.planning.simulation.path.utils import get_trimmed_path_up_to_progress
from shapely.geometry import CAP_STYLE, LineString, Polygon


@dataclass
class VelocityRecord:
    velocity: float  # [m/s^2], Velocity at the current timestamp
    timestamp: int  # timestamp
    distance_to_stop_line: float  # [m], Distance to the stop line


@dataclass
class VelocityData:
    data: List[VelocityRecord]

    def add_data(self,
                 velocity: float,
                 timestamp: int,
                 distance_to_stop_line: float
                 ) -> None:
        """
        Add new data to the list.
        :param velocity: [m/s^2], Velocity at the current timestamp.
        :param timestamp: Timestamp.
        :param distance_to_stop_line: [m], Distance to the stop line
        """

        if self.data is None:
            self.data = []

        self.data.append(
            VelocityRecord(
                velocity=velocity,
                timestamp=timestamp,
                distance_to_stop_line=distance_to_stop_line
            )
        )

    @property
    def velocity_np(self) -> npt.NDArray[np.float32]:
        """ Velocity in numpy representation. """

        return np.asarray([data.velocity for data in self.data])

    @property
    def timestamp_np(self) -> npt.NDArray[np.int32]:
        """ Timestamp in numpy representation. """

        return np.asarray([data.timestamp for data in self.data])

    @property
    def distance_to_stop_line_np(self) -> npt.NDArray[np.float32]:
        """ Distance to stop line in numpy representation. """

        return np.asarray([data.distance_to_stop_line for data in self.data])

    @property
    def min_distance_stop_line_record(self) -> VelocityRecord:
        """
        Return velocity record of minimum distance stop line.
        :return A velocity record.
        """

        distance_to_stop_line = self.distance_to_stop_line_np
        index = np.argmin(distance_to_stop_line)
        return self.data[int(index)]


class EgoStopAtStopLineStatistics(AbstractMetricBuilder):

    def __init__(self,
                 name: str,
                 category: str,
                 distance_threshold: float,
                 velocity_threshold: float) -> None:
        """
        Ego stopped at stop line metric.
        Rule formulation: 1. Get the nearest stop polygon (less than the distance threshold).
                          2. Check if the stop polygon is in any lanes.
                          3. Check if front corners of ego cross the stop polygon.
                          4. Check if no any leading agents.
                          5. Get min_velocity(distance_stop_line) until the ego leaves the stop polygon.
        :param name: Metric name.
        :param category: Metric category.
        :param distance_threshold: Distances between ego front side and stop line lower than this threshold
        assumed to be the first vehicle before the stop line.
        :param velocity_threshold: Velocity threshold to consider an ego stopped.
        """

        self._name = name
        self._category = category
        self._distance_threshold = distance_threshold
        self._velocity_threshold = velocity_threshold
        self._stopping_velocity_data: List[VelocityData] = []
        self._previous_stop_polygon_fid: Optional[str] = None

    @property
    def name(self) -> str:
        """
        Returns the metric name.
        :return: the metric name.
        """

        return self._name

    @property
    def category(self) -> str:
        """
        Returns the metric category.
        :return: the metric category.
        """

        return self._category

    @staticmethod
    def get_nearest_stop_line(map_api: AbstractMap,
                              ego_pose_front: LineString) -> Optional[Tuple[str, Polygon]]:
        """
        Retrieve the nearest stop polygon.
        :param map_api: AbstractMap map api.
        :param ego_pose_front: Ego pose front corner line.
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
    def check_for_leading_agents(detections: Observation,
                                 ego_box_3d: Box3D,
                                 map_api: AbstractMap) -> bool:
        """
        Get the nearest leading agent.
        :param detections: Detection class.
        :param ego_box_3d: Ego in 3d box representation.
        :param map_api: AbstractMap api.
        :return True if there is a leading agent, False otherwise
        """

        if isinstance(detections, Detections):

            # Assign a new instance token to ego pose
            ego_box_3d.token = '0'

            # Check if any missing instance token
            for index, box in enumerate(detections.boxes):
                if box.token is None:
                    box.token = str(index + 1)
            agent_boxes = [ego_box_3d] + detections.boxes

            if len(detections.boxes) == 0:
                return False

            occupancy_map = STRTreeOccupancyMapFactory.get_from_boxes(agent_boxes)
            agent_states = {box.token: StateSE2(x=box.center[0], y=box.center[1],
                                                heading=quaternion_yaw(box.orientation)) for box in agent_boxes}
            ego_state: StateSE2 = agent_states['0']
            lane = map_api.get_one_map_object(ego_state, SemanticMapLayer.LANE)

            # Construct ego's path to go
            ego_baseline = lane.baseline_path()
            ego_progress = ego_baseline.get_nearest_arc_length_from_position(ego_state)
            progress_path = create_path_from_se2(ego_baseline.discrete_path())
            ego_path_to_go = get_trimmed_path_up_to_progress(progress_path, ego_progress)

            # Check for intersection between the path and the any other agents
            ego_path_to_go = path_to_linestring(ego_path_to_go)
            intersecting_agents = occupancy_map.intersects(ego_path_to_go.buffer((agent_boxes[0].width / 2),
                                                                                 cap_style=CAP_STYLE.flat))

            # If there are other agents ahead of the ego in the same lane there should be at least two agents
            if intersecting_agents.size > 1:
                return True

        return False

    def _compute_velocity_statistics(self) -> List[MetricStatistics]:
        """
        Compute statistics in each stop line.
        :return A list of metric statistics.
        """

        results = []
        for velocity_data in self._stopping_velocity_data:

            # Stop at stop line success rate, we check the velocity of minimum distance to the stop line in each record
            min_distance_velocity_record = velocity_data.min_distance_stop_line_record
            if min_distance_velocity_record.distance_to_stop_line > self._distance_threshold:
                continue

            if min_distance_velocity_record.velocity < self._velocity_threshold:
                stop_status = True
            else:
                stop_status = False

            statistics = {
                MetricStatisticsType.DISTANCE: Statistic(name="ego_min_distance_to_stop_line", unit="meters",
                                                         value=min_distance_velocity_record.distance_to_stop_line),
                MetricStatisticsType.VELOCITY: Statistic(name="ego_min_velocity_before_stop_line",
                                                         unit="meters_per_second_squared",
                                                         value=min_distance_velocity_record.velocity),
                MetricStatisticsType.BOOLEAN: Statistic(name="ego_stop_status", unit="boolean", value=stop_status),
            }

            time_stamps = velocity_data.timestamp_np
            velocity_time_series = TimeSeries(unit='meters_per_second_squared',
                                              time_stamps=list(time_stamps),
                                              values=list(velocity_data.velocity_np))

            result = MetricStatistics(metric_computator=self.name,
                                      name="ego_stop_at_stop_line_statistics",
                                      statistics=statistics,
                                      time_series=velocity_time_series,
                                      metric_category=self.category)
            results.append(result)

        return results

    def _save_stopping_velocity(self,
                                current_stop_polygon_fid: str,
                                history_data: SimulationHistorySample,
                                stop_polygon_in_lane: Polygon,
                                ego_pose_front: LineString) -> None:
        """
        Save velocity, timestamp and distance to a stop line if the ego is stopping.
        :param current_stop_polygon_fid: Current stop polygon fid.
        :param history_data: History sample data at current timestamp.
        :param stop_polygon_in_lane: The stop polygon where the ego is in.
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
                distance_to_stop_line=distance_ego_front_stop_line
            )
        else:
            # If it is a new stop polygon
            self._previous_stop_polygon_fid = current_stop_polygon_fid
            velocity_data = VelocityData([])
            velocity_data.add_data(
                velocity=current_velocity,
                timestamp=current_timestamp,
                distance_to_stop_line=distance_ego_front_stop_line
            )
            self._stopping_velocity_data.append(velocity_data)

    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """

        # Extract ego pose in box3d.
        ego_pose_box_3ds: List[Box3D] = [ego_state_to_box_3d(sample.ego_state) for sample in history.data]

        # Get egos' front footprint
        ego_pose_fronts: List[LineString] = [LineString(box_3d.bottom_corners[:2, :2].T) for box_3d in ego_pose_box_3ds]

        # Get stop polygons
        scenario_map: AbstractMap = history.map_api

        # Get the nearest stop polygon.
        for ego_pose_front, ego_pose_box_3d, history_data in zip(ego_pose_fronts, ego_pose_box_3ds, history.data):

            # 1) Get the nearest stop polygon if it is in the same lane as ego.
            stop_polygon_info: Optional[Tuple[str, Polygon]] = self.get_nearest_stop_line(map_api=scenario_map,
                                                                                          ego_pose_front=ego_pose_front)

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
            has_leading_agent = self.check_for_leading_agents(detections=detections,
                                                              ego_box_3d=ego_pose_box_3d,
                                                              map_api=scenario_map)

            # Skip if there is any leading agent in the stop polygon (means the ego is not the first vehicle)
            if has_leading_agent:
                continue

            # 4) Accumulate velocity
            self._save_stopping_velocity(
                current_stop_polygon_fid=fid,
                history_data=history_data,
                stop_polygon_in_lane=stop_polygon_in_lane,
                ego_pose_front=ego_pose_front
            )

        results = self._compute_velocity_statistics()
        return results
