import logging
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent, IDMInitialState
from nuplan.planning.simulation.observation.idm.idm_agent_manager import UniqueIDMAgents
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import (
    STRTreeOccupancyMap,
    STRTreeOccupancyMapFactory,
)

logger = logging.getLogger(__name__)


def get_starting_segment(
    agent: Agent, map_api: AbstractMap
) -> Tuple[Optional[LaneGraphEdgeMapObject], Optional[float]]:
    """
    Gets the map object that the agent is on and the progress along the segment.
    :param agent: The agent of interested.
    :param map_api: An AbstractMap instance.
    :return: GraphEdgeMapObject and progress along the segment. If no map object is found then None.
    """
    if map_api.is_in_layer(agent.center, SemanticMapLayer.LANE):
        layer = SemanticMapLayer.LANE
    elif map_api.is_in_layer(agent.center, SemanticMapLayer.INTERSECTION):
        layer = SemanticMapLayer.LANE_CONNECTOR
    else:
        return None, None

    segments: List[LaneGraphEdgeMapObject] = map_api.get_all_map_objects(agent.center, layer)
    if not segments:
        return None, None

    # Get segment with the closest heading to the agent
    heading_diff = [
        segment.baseline_path.get_nearest_pose_from_position(agent.center).heading - agent.center.heading
        for segment in segments
    ]
    closest_segment = segments[np.argmin(np.abs(heading_diff))]

    progress = closest_segment.baseline_path.get_nearest_arc_length_from_position(agent.center)
    return closest_segment, progress


def build_idm_agents_on_map_rails(
    target_velocity: float,
    min_gap_to_lead_agent: float,
    headway_time: float,
    accel_max: float,
    decel_max: float,
    minimum_path_length: float,
    scenario: AbstractScenario,
    open_loop_detections_types: List[TrackedObjectType],
) -> Tuple[UniqueIDMAgents, OccupancyMap]:
    """
    Build unique agents from a scenario. InterpolatedPaths are created for each agent according to their driven path

    :param target_velocity: Desired velocity in free traffic [m/s]
    :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
    :param headway_time: Desired time headway. The minimum possible time to the vehicle in front [s]
    :param accel_max: maximum acceleration [m/s^2]
    :param decel_max: maximum deceleration (positive value) [m/s^2]
    :param minimum_path_length: [m] The minimum path length
    :param scenario: scenario
    :param open_loop_detections_types: The open-loop detection types to include.
    :return: a dictionary of IDM agent uniquely identified by a track_token
    """
    unique_agents: UniqueIDMAgents = {}

    detections = scenario.initial_tracked_objects
    map_api = scenario.map_api
    ego_agent = scenario.get_ego_state_at_iteration(0).agent

    open_loop_detections = detections.tracked_objects.get_tracked_objects_of_types(open_loop_detections_types)
    # An occupancy map used only for collision checking
    init_agent_occupancy = STRTreeOccupancyMapFactory.get_from_boxes(open_loop_detections)
    init_agent_occupancy.insert(ego_agent.token, ego_agent.box.geometry)

    # Initialize occupancy map
    occupancy_map = STRTreeOccupancyMap({})
    desc = "Converting detections to smart agents"

    agent: Agent
    for agent in tqdm(
        detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE), desc=desc, leave=False
    ):
        # filter for only vehicles
        if agent.track_token not in unique_agents:

            route, progress = get_starting_segment(agent, map_api)

            # Ignore agents that a baseline path cannot be built for
            if route is None:
                continue

            # Snap agent to baseline path
            state_on_path = route.baseline_path.get_nearest_pose_from_position(agent.center.point)
            box_on_baseline = OrientedBox.from_new_pose(
                agent.box, StateSE2(state_on_path.x, state_on_path.y, state_on_path.heading)
            )

            # Check for collision
            if not init_agent_occupancy.intersects(box_on_baseline.geometry).is_empty():
                continue

            # Add to init_agent_occupancy for collision checking
            init_agent_occupancy.insert(agent.track_token, box_on_baseline.geometry)

            # Add to occupancy_map to pass on to IDMAgentManger
            occupancy_map.insert(agent.track_token, box_on_baseline.geometry)

            # Project velocity into local frame
            if np.isnan(agent.velocity.array).any():
                ego_state = scenario.get_ego_state_at_iteration(0)
                logger.debug(
                    f"Agents has nan velocity. Setting velocity to ego's velocity of "
                    f"{ego_state.dynamic_car_state.speed}"
                )
                velocity = StateVector2D(ego_state.dynamic_car_state.speed, 0.0)
            else:
                velocity = StateVector2D(np.hypot(agent.velocity.x, agent.velocity.y), 0)

            initial_state = IDMInitialState(
                metadata=agent.metadata,
                tracked_object_type=agent.tracked_object_type,
                box=box_on_baseline,
                velocity=velocity,
                path_progress=progress,
                predictions=agent.predictions,
            )
            target_velocity = route.speed_limit_mps or target_velocity
            unique_agents[agent.track_token] = IDMAgent(
                start_iteration=0,
                initial_state=initial_state,
                route=[route],
                policy=IDMPolicy(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max),
                minimum_path_length=minimum_path_length,
            )

    return unique_agents, occupancy_map
