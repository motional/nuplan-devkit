import logging
from abc import ABC
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import CAP_STYLE
from shapely.ops import unary_union

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.transform import transform
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.observation.idm.idm_states import IDMAgentState, IDMLeadAgentState
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from nuplan.planning.simulation.path.path import AbstractPath
from nuplan.planning.simulation.path.utils import trim_path
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

UniqueObjects = Dict[str, SceneObject]

logger = logging.getLogger(__name__)


class AbstractIDMPlanner(AbstractPlanner, ABC):
    """
    An interface for IDM based planners. Inherit from this class to use IDM policy to control the longitudinal
    behaviour of the ego.
    """

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
        planned_trajectory_samples: int,
        planned_trajectory_sample_interval: float,
        occupancy_map_radius: float,
    ):
        """
        Constructor for IDMPlanner
        :param target_velocity: [m/s] Desired velocity in free traffic.
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle.
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front.
        :param accel_max: [m/s^2] maximum acceleration.
        :param decel_max: [m/s^2] maximum deceleration (positive value).
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param occupancy_map_radius: [m] The range around the ego to add objects to be considered.
        """
        self._policy = IDMPolicy(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max)
        self._planned_trajectory_samples = planned_trajectory_samples
        self._planned_trajectory_sample_interval = planned_trajectory_sample_interval
        self._planned_horizon = planned_trajectory_samples * planned_trajectory_sample_interval
        self._occupancy_map_radius = occupancy_map_radius
        self._max_path_length = self._policy.target_velocity * self._planned_horizon
        self._ego_token = "ego_token"
        self._red_light_token = "red_light"

        # To be lazy loaded
        self._route_roadblocks: List[RoadBlockGraphEdgeMapObject] = []
        self._candidate_lane_edge_ids: Optional[List[str]] = None
        self._map_api: Optional[AbstractMap] = None

        # To be intialized by inherited classes
        self._ego_path: Optional[AbstractPath] = None
        self._ego_path_linestring: Optional[LineString] = None

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def _initialize_route_plan(self, route_roadblock_ids: List[str]) -> None:
        """
        Initializes the route plan with roadblocks.
        :param route_roadblock_ids: A list of roadblock ids that make up the ego's route
        """
        assert self._map_api, "_map_api has not yet been initialized. Please call the initialize() function first!"
        self._route_roadblocks = []
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

        assert (
            self._route_roadblocks
        ), "Cannot create route plan. No roadblocks were extracted from the given route_roadblock_ids!"

    def _get_expanded_ego_path(self, ego_state: EgoState, ego_idm_state: IDMAgentState) -> Polygon:
        """
        Returns the ego's expanded path as a Polygon.
        :return: A polygon representing the ego's path.
        """
        assert self._ego_path, "_ego_path has not yet been initialized. Please call the initialize() function first!"
        ego_footprint = ego_state.car_footprint
        path_to_go = trim_path(
            self._ego_path,
            max(self._ego_path.get_start_progress(), min(ego_idm_state.progress, self._ego_path.get_end_progress())),
            max(
                self._ego_path.get_start_progress(),
                min(
                    ego_idm_state.progress + abs(self._policy.target_velocity) * self._planned_horizon,
                    self._ego_path.get_end_progress(),
                ),
            ),
        )
        expanded_path = path_to_linestring(path_to_go).buffer((ego_footprint.width / 2), cap_style=CAP_STYLE.square)
        return unary_union([expanded_path, ego_state.car_footprint.geometry])

    @staticmethod
    def _get_leading_idm_agent(ego_state: EgoState, agent: SceneObject, relative_distance: float) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state that represents another static and dynamic agent.
        :param agent: A scene object.
        :param relative_distance: [m] The relative distance from the scene object to the ego.
        :return: A IDM lead agents state
        """
        if isinstance(agent, Agent):
            # Dynamic object
            longitudinal_velocity = agent.velocity.magnitude()
            # Wrap angle to [-pi, pi]
            relative_heading = principal_value(agent.center.heading - ego_state.center.heading)
            projected_velocity = transform(
                StateSE2(longitudinal_velocity, 0, 0), StateSE2(0, 0, relative_heading).as_matrix()
            ).x
        else:
            # Static object
            projected_velocity = 0.0

        return IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=0.0)

    def _get_free_road_leading_idm_state(self, ego_state: EgoState, ego_idm_state: IDMAgentState) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state when there is no leading agent.
        :return: A IDM lead agents state.
        """
        assert self._ego_path, "_ego_path has not yet been initialized. Please call the initialize() function first!"
        projected_velocity = 0.0
        relative_distance = self._ego_path.get_end_progress() - ego_idm_state.progress
        length_rear = ego_state.car_footprint.length / 2
        return IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=length_rear)

    @staticmethod
    def _get_red_light_leading_idm_state(relative_distance: float) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state that represents a red light intersection.
        :param relative_distance: [m] The relative distance from the intersection to the ego.
        :return: A IDM lead agents state.
        """
        return IDMLeadAgentState(progress=relative_distance, velocity=0, length_rear=0)

    def _get_leading_object(
        self,
        ego_idm_state: IDMAgentState,
        ego_state: EgoState,
        occupancy_map: OccupancyMap,
        unique_observations: UniqueObjects,
    ) -> IDMLeadAgentState:
        """
        Get the most suitable leading object based on the occupancy map.
        :param ego_idm_state: The ego's IDM state at current iteration.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        """
        intersecting_agents = occupancy_map.intersects(self._get_expanded_ego_path(ego_state, ego_idm_state))
        # Check if there are agents intersecting the ego's baseline
        if intersecting_agents.size > 0:

            # Extract closest object
            intersecting_agents.insert(self._ego_token, ego_state.car_footprint.geometry)
            nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(
                self._ego_token
            )

            # Red light at intersection
            if self._red_light_token in nearest_id:
                return self._get_red_light_leading_idm_state(relative_distance)

            # An agent is the leading agent
            return self._get_leading_idm_agent(ego_state, unique_observations[nearest_id], relative_distance)

        else:
            # No leading agent
            return self._get_free_road_leading_idm_state(ego_state, ego_idm_state)

    def _construct_occupancy_map(
        self, ego_state: EgoState, observation: Observation
    ) -> Tuple[OccupancyMap, UniqueObjects]:
        """
        Constructs an OccupancyMap from Observations.
        :param ego_state: Current EgoState
        :param observation: Observations of other agents and static objects in the scene.
        :return:
            - OccupancyMap.
            - A mapping between the object token and the object itself.
        """
        if isinstance(observation, DetectionsTracks):
            unique_observations = {
                detection.track_token: detection
                for detection in observation.tracked_objects.tracked_objects
                if np.linalg.norm(ego_state.center.array - detection.center.array) < self._occupancy_map_radius
            }
            return (
                STRTreeOccupancyMapFactory.get_from_boxes(list(unique_observations.values())),
                unique_observations,
            )
        else:
            raise ValueError(f"IDM planner only supports DetectionsTracks. Got {observation.detection_type()}")

    def _propagate(self, ego: IDMAgentState, lead_agent: IDMLeadAgentState, tspan: float) -> None:
        """
        Propagate agent forward according to the IDM policy.
        :param ego: The ego's IDM state.
        :param lead_agent: The agent leading this agent.
        :param tspan: [s] The interval of time to propagate for.
        """
        # TODO: Set target velocity to speed limit
        solution = self._policy.solve_forward_euler_idm_policy(IDMAgentState(0, ego.velocity), lead_agent, tspan)
        ego.progress += solution.progress
        ego.velocity = max(solution.velocity, 0)

    def _get_planned_trajectory(
        self, ego_state: EgoState, occupancy_map: OccupancyMap, unique_observations: UniqueObjects
    ) -> InterpolatedTrajectory:
        """
        Plan a trajectory w.r.t. the occupancy map.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        :return: A trajectory representing the predicted ego's position in future.
        """
        assert (
            self._ego_path_linestring
        ), "_ego_path_linestring has not yet been initialized. Please call the initialize() function first!"
        # Extract ego IDM state
        ego_progress = self._ego_path_linestring.project(Point(*ego_state.center.point.array))
        ego_idm_state = IDMAgentState(progress=ego_progress, velocity=ego_state.dynamic_car_state.center_velocity_2d.x)
        vehicle_parameters = ego_state.car_footprint.vehicle_parameters

        # Initialize planned trajectory with current state
        current_time_point = ego_state.time_point
        projected_ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)
        planned_trajectory: List[EgoState] = [projected_ego_state]

        # Propagate planned trajectory for set number of samples
        for _ in range(self._planned_trajectory_samples):

            # Propagate IDM state w.r.t. selected leading agent
            leading_agent = self._get_leading_object(ego_idm_state, ego_state, occupancy_map, unique_observations)
            self._propagate(ego_idm_state, leading_agent, self._planned_trajectory_sample_interval)

            # Convert IDM state back to EgoState
            current_time_point += TimePoint(int(self._planned_trajectory_sample_interval * 1e6))
            ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)

            planned_trajectory.append(ego_state)

        return InterpolatedTrajectory(planned_trajectory)

    def _idm_state_to_ego_state(
        self, idm_state: IDMAgentState, time_point: TimePoint, vehicle_parameters: VehicleParameters
    ) -> EgoState:
        """
        Convert IDMAgentState to EgoState
        :param idm_state: The IDMAgentState to be converted.
        :param time_point: The TimePoint corresponding to the state.
        :param vehicle_parameters: VehicleParameters of the ego.
        """
        assert self._ego_path, "_ego_path has not yet been initialized. Please call the initialize() function first!"

        new_ego_center = self._ego_path.get_state_at_progress(
            max(self._ego_path.get_start_progress(), min(idm_state.progress, self._ego_path.get_end_progress()))
        )
        return EgoState.build_from_center(
            center=StateSE2(new_ego_center.x, new_ego_center.y, new_ego_center.heading),
            center_velocity_2d=StateVector2D(idm_state.velocity, 0),
            center_acceleration_2d=StateVector2D(0, 0),
            tire_steering_angle=0.0,
            time_point=time_point,
            vehicle_parameters=vehicle_parameters,
        )

    def _annotate_occupancy_map(
        self, traffic_light_data: List[TrafficLightStatusData], occupancy_map: OccupancyMap
    ) -> None:
        """
        Add red light lane connectors on the route plan to the occupancy map. Note: the function works inline, hence,
        the occupancy map will be modified in this function.
        :param traffic_light_data: A list of all available traffic status data.
        :param occupancy_map: The occupancy map to be annotated.
        """
        assert self._map_api, "_map_api has not yet been initialized. Please call the initialize() function first!"
        assert (
            self._candidate_lane_edge_ids is not None
        ), "_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!"
        for data in traffic_light_data:
            if (
                data.status == TrafficLightStatusType.RED
                and str(data.lane_connector_id) in self._candidate_lane_edge_ids
            ):
                id_ = str(data.lane_connector_id)
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                occupancy_map.insert(f"{self._red_light_token}_{id_}", lane_conn.polygon)
