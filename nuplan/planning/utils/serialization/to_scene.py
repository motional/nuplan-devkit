import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
from nuplan.common.actor_state.agent import AgentType
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.transform_state import translate_longitudinally
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.label.utils import LabelMapping, local2agent_type
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory
from pyquaternion import Quaternion


def to_scene_from_states(trajectory: List[EgoState]) -> List[Dict[str, Any]]:
    """
    Convert trajectory into structure
    :param trajectory: input trajectory ot be converted
    :return scene
    """
    return [{"pose": [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading],
             "speed": state.dynamic_car_state.speed,
             "velocity_2d": [state.dynamic_car_state.rear_axle_velocity_2d.x,
                             state.dynamic_car_state.rear_axle_velocity_2d.y],
             "lateral": [0.0, 0.0], "acceleration": [state.dynamic_car_state.rear_axle_acceleration_2d.x,
                                                     state.dynamic_car_state.rear_axle_acceleration_2d.y],
             "tire_steering_angle": state.tire_steering_angle} for state in
            trajectory]


def to_scene_from_trajectory(trajectory: AbstractTrajectory) -> List[Dict[str, Any]]:
    """
    Convert trajectory into structure
    :param trajectory: input trajectory ot be converted
    :return scene
    """
    return to_scene_from_states(trajectory.get_sampled_trajectory())


def create_trajectory_structure(trajectory: List[EgoState], color: List[int]) -> Dict[str, Any]:
    """
    Create scene json structure for a trajectory with a color
    :rtype: object
    :param trajectory: set of states
    :param color: color [R, G, B, A]
    :return: scene representing a trajectory
    """
    return {"color": color,
            "states": to_scene_from_states(trajectory)}


def to_scene_ego_from_center_pose(state_center: StateSE2, vehicle: VehicleParameters) -> Dict[str, Any]:
    """
    :param state_center: state of ego's center pose
    :param vehicle: vehicle parameters
    :return scene
    """

    distance = -vehicle.rear_axle_to_center
    ego_center_x, ego_center_y = translate_longitudinally(state_center, distance)
    return {
        "acceleration": 0.0,
        "pose": [ego_center_x, ego_center_y, state_center.heading],
        "speed": 0.0
    }


def to_scene_agent_type(agent_type: AgentType) -> str:
    """
    AgentType to string
    :param agent_type: AgentType
    :return string representing an agent type
    """
    if agent_type == AgentType.VEHICLE:
        return "Vehicle"
    elif agent_type == AgentType.PEDESTRIAN:
        return "Pedestrian"
    elif agent_type == AgentType.BICYCLE:
        return "Bicycle"
    raise ValueError("Unknown input type: {}".format(str(agent_type)))


def to_scene_box(box: Box3D, track_id: int, agent_type: AgentType) -> Dict[str, Any]:
    """
    Convert box into json representation
    :param box: 3D box representation
    :param track_id: unique id of an agent
    :param agent_type: type of an agent
    :return json representation of an agent
    """
    center_x = box.center[0]
    center_y = box.center[1]
    center_heading = box.yaw
    speed = np.hypot(box.velocity[0], box.velocity[1])
    scene = {
        "active": True,
        "real": True,
        "speed": speed if not np.isnan(speed) else 0.0,
        "box": {
            "pose": [center_x, center_y, center_heading],
            "size": [box.width, box.length]
        },
        "id": track_id,
        "type": to_scene_agent_type(agent_type)
    }
    if agent_type == AgentType.PEDESTRIAN:
        scene["box"]["radius"] = 0.5
    return scene


def to_scene_boxes(boxes: List[Box3D], labelmap: LabelMapping) -> Dict[str, Any]:
    """
    Convert boxes into a scene
    :param boxes: List of boxes in global coordinates
    :param labelmap: convert label into agent type
    :return dictionary which should be placed into scene["world"]
    """
    id2local = {value: key for key, value in labelmap.local2id.items()}

    def generate_track(agent_type: AgentType, index_offset: int) -> List[Dict[str, Any]]:
        """
        Convert boxes af type into scene
        :param agent_type: type of an agent
        :param index_offset: offset of track ids from initial box
        :return dict of tracks
        """
        return [to_scene_box(box, index_offset + i, agent_type) for i, box in enumerate(boxes) if
                AgentType[local2agent_type[id2local[box.label]]] == agent_type]

    vehicles = generate_track(AgentType.VEHICLE, 0)
    pedestrians = generate_track(AgentType.PEDESTRIAN, len(vehicles))
    bicycles = generate_track(AgentType.BICYCLE, len(vehicles) + len(pedestrians))

    return {
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "bicycles": bicycles,
    }


def to_scene_from_ego_and_boxes(ego_pose: StateSE2, boxes: List[Box3D], map_name: str, labelmap: LabelMapping,
                                set_camera: bool = False, vehicle: VehicleParameters = get_pacifica_parameters()) \
        -> Dict[str, Any]:
    """
    Extract scene from ego_pose and boxes
    :param ego_pose: Ego Position
    :param boxes: list of actors in global coordinate frame
    :param map_name: map name
    :param labelmap: mapping box labels to types
    :param set_camera: True if we wish to also set camera view
    :param vehicle: VehicleParameters
    :return scene
    """
    # World
    world = to_scene_boxes(boxes, labelmap)

    # Extract map name
    map_name_without_suffix = str(pathlib.Path(map_name).stem)

    # Construct Scene
    scene = {
        "map": {"area": map_name_without_suffix},
        "world": world,
        "ego": to_scene_ego_from_center_pose(ego_pose, vehicle)
    }

    if set_camera:
        ego_pose = scene["ego"]["pose"]
        ego_x = ego_pose[0]
        ego_y = ego_pose[1]
        ego_heading = ego_pose[2]
        bearing_rad = np.fmod(ego_heading, np.pi * 2)  # In radians
        if bearing_rad < 0:
            bearing_rad += np.pi * 2
        bearing_rad = 1.75 - (bearing_rad / (np.pi * 2))
        if bearing_rad >= 1:
            bearing_rad -= 1.0
        scene["camera"] = {
            "pitch": 50,
            "scale": 2500000,
            "bearing": bearing_rad,
            "lookat": [ego_x, ego_y, 0.0]
        }

    return scene


def _to_scene_agent_prediction(box: Box3D, color: List[int], box_id: int) -> Dict[str, Any]:
    """
    Extract agent's predicted states from Box3D to scene
    :param box: 3D box representation
    :param color: color [R, G, B, A]
    :param box_id: unique id for each agent
    :return a prediction scene
    """

    def extract_prediction_state(center: Tuple[float, float, float], orientation: Quaternion) -> Dict[str, Any]:
        return {
            "pose": [center[0], center[1], orientation.yaw_pitch_roll[0]],
            "polygon": [[center[0], center[1]]]
        }

    return {
        "id": box_id,
        "color": color,
        "states": [extract_prediction_state(center, orientation)
                   for modes in range(box.mode_probs.shape[0])
                   for center, orientation in
                   zip(box.future_centers[modes], box.future_orientations[modes])]
    }


def to_scene_agent_prediction_from_boxes(boxes: List[Box3D], color: List[int]) \
        -> List[Dict[str, Any]]:
    """
    Convert predicted observations into prediction dictionary

    :param boxes: List of boxes in global coordinates
    :param color: color [R, G, B, A]
    :return scene
    """
    return [_to_scene_agent_prediction(agent, color, agent_id) for agent_id, agent in enumerate(boxes)
            if agent.future_centers is not None]
