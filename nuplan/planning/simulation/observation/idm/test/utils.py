from typing import Any, Dict, List

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent, IDMInitialState
from nuplan.planning.simulation.observation.idm.idm_agent_manager import IDMAgentManager, UniqueIDMAgents
from nuplan.planning.simulation.observation.idm.idm_agents_builder import get_starting_segment
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from nuplan.planning.utils.serialization.from_scene import from_scene_to_tracked_objects


def baseline_path_to_scene(scene: Dict[str, Any], baseline_path: List[StateSE2]) -> None:
    """
    Renders an agent's reference baseline path as a series of poses
    :param scene: scene dictionary
    :param baseline_path: baseline path represented by a list fo StateSE2
    """
    if "path_info" not in scene.keys():
        scene["path_info"] = []

    scene["path_info"] += [[pose.x, pose.y, pose.heading] for pose in baseline_path]


def baseline_path_as_shapes_to_scene(scene: Dict[str, Any], agents: UniqueIDMAgents) -> None:
    """
    Renders all agents' reference baseline paths as a colored shapes
    :param scene: scene dictionary
    :param agents: all agents for which the baseline path should be rendered for
    """
    if "shapes" not in scene.keys():
        scene["shapes"] = dict()

    colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]

    color_inc = 0
    for agent_id, agent in agents.items():
        cut_path = agent.projected_footprint
        scene["shapes"][str(agent_id)] = {
            "color": colors[color_inc],
            "filled": True,
            "objects": [[[x, y] for x, y in zip(*cut_path.exterior.xy)]],
        }
        color_inc = (color_inc + 1) % len(colors)


def agent_baselines_to_scene(scene: Dict[str, Any], agents: UniqueIDMAgents) -> None:
    """
    Renders all agents' reference baseline paths as series of poses
    :param scene: scene dictionary
    :param agents: all agents for which the baseline path should be rendered for
    """
    for agent in agents.values():
        baseline_path_to_scene(scene, agent.get_path_to_go())


def marker_to_scene(scene: Dict[str, Any], marker_id: str, pose: StateSE2) -> None:
    """
    Renders a pose as an arrow marker
    :param scene: scene dictionary
    :param marker_id: marker id as a string
    :param pose: the pose that defines the markers location
    """
    if "markers" not in scene.keys():
        scene["markers"] = []

    scene["markers"].append({"id": 0, "name": marker_id, "pose": pose.serialize(), "shape": "arrow"})


def agents_pose_to_scene(scene: Dict[str, Any], agents: UniqueIDMAgents) -> None:
    """
    Renders all agents' pose as markers
    :param scene: scene dictionary
    :param agents: all agents for which the pose should be rendered for
    """
    for agent_id, agent in agents.items():
        marker_to_scene(scene, agent_id, agent.to_se2())


def agents_box_to_scene(scene: Dict[str, Any], agents: UniqueIDMAgents) -> None:
    """
    Renders all agents' box in scene
    :param scene: scene dictionary
    :param agents: all agents to be rendered
    """
    for vehicle in scene["world"]["vehicles"]:
        agent_id = str(vehicle["id"])
        if agent_id in agents:
            vehicle["box"]["pose"] = agents[agent_id].to_se2().serialize()
            vehicle["speed"] = agents[agent_id].velocity


def agent_state_to_scene(scene: Dict[str, Any], agent: EgoState, agent_id: int) -> None:
    """
    Renders all agent_state as box in scene
    :param scene: scene dictionary
    :param agent: agent state
    :param agent_id: an unique id
    """
    scene["world"]["vehicles"].append(
        {
            "active": True,
            "box": {
                "pose": [agent.center.x, agent.center.y, agent.center.heading],
                "size": [agent.car_footprint.oriented_box.width, agent.car_footprint.oriented_box.length],
            },
            "id": agent_id,
            "real": True,
            "speed": agent.dynamic_car_state.rear_axle_velocity_2d.x,
            "type": "Vehicle",
        }
    )


def _build_idm_agents(agents: List[Agent], map_api: AbstractMap, policy: IDMPolicy) -> UniqueIDMAgents:
    """
    Builds idm agents.
    :param agents: list of agents represented by Agent
    :param map_api: AbstractMap
    :param policy: IDM policy
    :return: A dictionary of unique agents
    """
    unique_agents: UniqueIDMAgents = {}

    for agent in agents:
        route, progress = get_starting_segment(agent, map_api)
        initial_state = IDMInitialState(
            metadata=agent.metadata,
            tracked_object_type=agent.tracked_object_type,
            box=agent.box,
            velocity=agent.velocity,
            path_progress=progress,
            predictions=None,
        )
        unique_agents[str(agent.token)] = IDMAgent(
            start_iteration=0, initial_state=initial_state, route=[route], policy=policy, minimum_path_length=10
        )

    return unique_agents


def build_idm_manager(scene: Dict[str, Any], map_factory: AbstractMapFactory, policy: IDMPolicy) -> IDMAgentManager:
    """
    Builds IDMAgentManager from scene
    :param scene: scene dictionary
    :param map_factory: AbstractMapFactory
    :param policy: IDM policy
    :return: IDMAgentManager object
    """
    map_name = scene["map"]["area"]
    map_maps_db = map_factory.build_map_from_name(map_name)
    agents = from_scene_to_tracked_objects(scene["world"]).get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
    unique_agents = _build_idm_agents(agents, map_maps_db, policy)
    occupancy_map = STRTreeOccupancyMapFactory().get_from_boxes(agents)
    idm_manager = IDMAgentManager(unique_agents, occupancy_map, map_maps_db)

    return idm_manager
