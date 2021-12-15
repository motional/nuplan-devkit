from typing import Any, Dict, List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_agent_manager import UniqueIDMAgents
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import ego_state_to_box_3d, path_to_linestring


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

    colors = [[1.0, 0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0, 1.0],
              [1.0, 0.0, 1.0, 1.0],
              [1.0, 1.0, 0.0, 1.0]]

    color_inc = 0
    for agent_id, agent in agents.items():
        cut_path = path_to_linestring(agent.get_path_to_go()).buffer(agent.width / 2, cap_style=2)
        scene["shapes"][str(agent_id)] = {
            "color": colors[color_inc],
            "filled": True,
            "objects": [[[x, y] for x, y in zip(*cut_path.exterior.xy)]]
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

    scene["markers"].append({
        "id": 0,
        "name": marker_id,
        "pose": pose.serialize(),
        "shape": "arrow"
    })


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
        vehicle["box"]["pose"] = agents[agent_id].to_se2().serialize()
        vehicle["speed"] = agents[agent_id].velocity


def agent_state_to_scene(scene: Dict[str, Any], agent: EgoState, agent_id: int) -> None:
    """
    Renders all agent_state as box in scene
    :param scene: scene dictionary
    :param agent: agent state
    :param agent_id: an unique id
    """
    ego_box = ego_state_to_box_3d(agent)
    scene["world"]["vehicles"].append({
        "active": True,
        "box": {
            "pose": [ego_box.center[0], ego_box.center[1], ego_box.yaw],
            "size": [ego_box.size[0], ego_box.size[1]]
        },
        "id": agent_id,
        "real": True,
        "speed": ego_box.velocity[0],
        "type": "Vehicle"
    })
