from typing import Any, Dict, List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.transform_state import translate_longitudinally_se2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.database.utils.boxes.box3d import Box3D
from pyquaternion import Quaternion


def to_state_from_scene(scene: Dict[str, Any]) -> StateSE2:
    """
    Extract state se2 from pose
    :param scene: position from scene
    :return StateSE2
    """
    return StateSE2(x=scene["pose"][0], y=scene["pose"][1], heading=scene["pose"][2])


def to_ego_center_from_scene(scene: Dict[str, Any], vehicle: VehicleParameters) -> StateSE2:
    """
    :param scene: from scene["ego"]
    :param vehicle: vehicle parameters
    :return the extracted State in the center of ego's bounding box
    """
    ego_pose = scene["pose"]

    ego_x = ego_pose[0]
    ego_y = ego_pose[1]
    ego_heading = ego_pose[2]
    distance = vehicle.rear_axle_to_center
    return translate_longitudinally_se2(StateSE2(ego_x, ego_y, ego_heading), distance)


def to_agent_state_from_scene(scene: Dict[str, Any],
                              vehicle: VehicleParameters,
                              time_us: int = 10,
                              to_cog: bool = True) -> EgoState:
    """
    Extract agent state from scene
    :param scene: from json
    :param vehicle: parameters
    :param time_us: [us] initial time
    :param to_cog: If true, xy will be translated to the COG of the car. Otherwise xy is assumed to be at rear axle
    :return EgoState
    """
    if to_cog:
        ego_state2d = to_ego_center_from_scene(scene, vehicle)
    else:
        ego_pose = scene["pose"]
        ego_state2d = StateSE2(ego_pose[0], ego_pose[1], ego_pose[2])

    # x-directions: longitudinal, y-directions: lateral.
    if "velocity_x" not in scene or "velocity_y" not in scene:
        velocity_x = scene['speed']
        velocity_y = 0.0
    else:
        velocity_x = scene["velocity_x"]
        velocity_y = scene["velocity_y"]

    if "acceleration_x" not in scene or "acceleration_y" not in scene:
        acceleration_x = scene['acceleration']
        acceleration_y = 0.0
    else:
        acceleration_x = scene["acceleration_x"]
        acceleration_y = scene["acceleration_y"]

    return EgoState.from_raw_params(StateSE2(x=ego_state2d.x,
                                             y=ego_state2d.y,
                                             heading=ego_state2d.heading),
                                    time_point=TimePoint(time_us),
                                    velocity_2d=StateVector2D(x=velocity_x, y=velocity_y),
                                    acceleration_2d=StateVector2D(x=acceleration_x, y=acceleration_y),
                                    tire_steering_angle=0)


def from_scene_box3d(scene: Dict[str, Any], label: int = 0) -> Box3D:
    """
    Convert scene to a Box3D
    :param scene: scene of an agent
    :param label: label of the resulting box
    :return Box3D extracted from a scene
    """
    box = scene["box"]
    pose = box["pose"]
    size = box["size"] if "size" in box.keys() else [0.5, 0.5]
    default_height = 1.5
    return Box3D(
        center=(pose[0], pose[1], 0.0),
        size=(size[0], size[1], default_height),
        orientation=Quaternion(axis=(0, 0, 1), angle=pose[2]),
        label=label,
        velocity=(scene["speed"], 0, 0)
    )


def from_scene_to_agents(scene: Dict[str, Any]) -> List[Box3D]:
    """
    Convert scene["world"] into boxes
    :param scene: scene["world"] coming from json
    :return List of boxes representing all agents
    """
    boxes: List[Box3D] = []
    boxes = boxes + [from_scene_box3d(track, 1) for track in scene["vehicles"]]
    boxes = boxes + [from_scene_box3d(track, 2) for track in scene["bicycles"]]
    boxes = boxes + [from_scene_box3d(track, 3) for track in scene["pedestrians"]]
    return boxes
