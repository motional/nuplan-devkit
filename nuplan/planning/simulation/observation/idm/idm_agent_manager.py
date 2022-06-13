from typing import Dict, List

import numpy as np
from shapely.geometry.base import CAP_STYLE

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.geometry.transform import rotate_angle
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import StopLine
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent
from nuplan.planning.simulation.observation.idm.idm_states import IDMLeadAgentState
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap

UniqueIDMAgents = Dict[str, IDMAgent]


class IDMAgentManager:
    """IDM smart-agents manager."""

    def __init__(self, agents: UniqueIDMAgents, agent_occupancy: OccupancyMap, map_api: AbstractMap):
        """
        Constructor for IDMAgentManager.
        :param agents: A dictionary pairing the agent's token to it's IDM representation.
        :param agent_occupancy: An occupancy map describing the spatial relationship between agents.
        :param map_api: AbstractMap API
        """
        self.agents: UniqueIDMAgents = agents
        self.agent_occupancy = agent_occupancy
        self._map_api = map_api

    def propagate_agents(
        self,
        ego_state: EgoState,
        tspan: float,
        iteration: int,
        traffic_light_status: Dict[TrafficLightStatusType, List[str]],
    ) -> None:
        """
        Propagate each active agent forward in time.

        :param ego_state: the ego's current state in the simulation.
        :param tspan: the interval of time to simulate.
        :param iteration: the simulation iteration.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        """
        self.agent_occupancy.set("ego", ego_state.car_footprint.geometry)

        # Add stop lines into occupancy map if they have red light status.

        stop_lines = self._get_stop_line_with_status(traffic_light_status, TrafficLightStatusType.RED)
        # Keep track of the stop lines that were inserted. This is to remove them after all agents have been propagated.
        inactive_stop_line_tokens = self._insert_stop_lines_into_occupancy_map(stop_lines)

        for agent_token, agent in self.agents.items():
            if agent.is_active(iteration) and agent.has_valid_path():
                agent.plan_route(traffic_light_status)

                # Check for agents that intersects THIS agent's path
                agent_path = path_to_linestring(agent.get_path_to_go())
                intersecting_agents = self.agent_occupancy.intersects(
                    agent_path.buffer((agent.width / 2), cap_style=CAP_STYLE.flat)
                )
                assert intersecting_agents.contains(agent_token), "Agent's baseline does not intersect the agent itself"

                # Checking if there are agents intersecting THIS agent's baseline.
                # Hence, we are checking for at least 2 intersecting agents.
                if intersecting_agents.size > 1:
                    nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(
                        agent_token
                    )
                    agent_heading = agent.to_se2().heading

                    if "ego" in nearest_id:
                        ego_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
                        longitudinal_velocity = np.hypot(ego_velocity.x, ego_velocity.y)
                        relative_heading = ego_state.rear_axle.heading - agent_heading
                    elif 'stop_line' in nearest_id:
                        longitudinal_velocity = 0.0
                        relative_heading = 0.0
                    else:
                        nearest_agent = self.agents[nearest_id]
                        longitudinal_velocity = nearest_agent.velocity
                        relative_heading = nearest_agent.to_se2().heading - agent_heading

                    # Wrap angle to [-pi, pi]
                    relative_heading = principal_value(relative_heading)
                    # take the longitudinal component of the projected velocity
                    projected_velocity = rotate_angle(StateSE2(longitudinal_velocity, 0, 0), relative_heading).x

                    # relative_distance already takes the vehicle dimension into account.
                    # Therefore there is no need to pass in the length_rear.
                    length_rear = 0
                else:
                    # Free road case: no leading vehicle
                    projected_velocity = 0.0
                    relative_distance = agent.get_progress_to_go()
                    length_rear = agent.length / 2

                agent.propagate(
                    IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=length_rear),
                    tspan,
                )
                self.agent_occupancy.set(agent_token, agent.projected_footprint)
        self.agent_occupancy.remove(inactive_stop_line_tokens)

    def get_active_agents(self, iteration: int, num_samples: int, sampling_time: float) -> DetectionsTracks:
        """
        Returns all agents as DetectionsTracks.
        :param iteration: the current simulation iteration.
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: agents as DetectionsTracks.
        """
        return DetectionsTracks(
            TrackedObjects(
                [
                    agent.get_agent_with_planned_trajectory(num_samples, sampling_time)
                    for agent in self.agents.values()
                    if agent.is_active(iteration)
                ]
            )
        )

    def _get_stop_line_with_status(
        self, traffic_light_status: Dict[TrafficLightStatusType, List[str]], status: TrafficLightStatusType
    ) -> List[StopLine]:
        """
        Retrieve the stop lines that has is labelled with the given TrafficLightStatusType.
        :param traffic_light_status: {traffic_light_status: lane_connector_ids} A dictionary containing traffic light information.
        :param status: Traffic light status.
        :return: A list of stop lines associated with the given traffic light status.
        """
        lane_connector_ids = traffic_light_status[status]
        lane_connectors = [
            self._map_api.get_map_object(lc_id, SemanticMapLayer.LANE_CONNECTOR) for lc_id in lane_connector_ids
        ]
        stop_lines = [stop_line for lc in lane_connectors if lc for stop_line in lc.get_stop_lines()]
        return stop_lines

    def _insert_stop_lines_into_occupancy_map(self, stop_lines: List[StopLine]) -> List[str]:
        """
        Insert stop lines into the occupancy map.
        :param stop_lines: A list of stop lines to be inserted.
        :return: A list of token corresponding to the inserted stop lines.
        """
        stop_line_tokens: List[str] = []
        for stop_line in stop_lines:
            stop_line_token = f"stop_line_{stop_line.id}"
            if not self.agent_occupancy.contains(stop_line_token):
                self.agent_occupancy.set(stop_line_token, stop_line.polygon)
                stop_line_tokens.append(stop_line_token)

        return stop_line_tokens
