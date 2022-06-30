import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.utils.color import Color

tracked_object_types = {
    'vehicles': TrackedObjectType.VEHICLE,
    'pedestrians': TrackedObjectType.PEDESTRIAN,
    'bicycles': TrackedObjectType.BICYCLE,
    'genericobjects': TrackedObjectType.GENERIC_OBJECT,
    'traffic_cone': TrackedObjectType.TRAFFIC_CONE,
    'barrier': TrackedObjectType.BARRIER,
    'czone_sign': TrackedObjectType.CZONE_SIGN,
}


def to_scene_ego_pose(ego_pose: EgoState, ego_future: Optional[List[EgoState]]) -> Dict[str, Any]:
    """
    :param ego_pose: state of ego
    :param ego_future: future ego position
    :return serialized scene
    """
    predictions = (
        {}
        if not ego_future
        else {
            "color": Color(red=0.498, green=0.498, blue=0.498, alpha=1.0).to_list(),
            "states": [
                {
                    "pose": [state.center.x, state.center.y, state.center.heading],
                    "timestamp": state.time_point.time_s - ego_pose.time_point.time_s,
                }
                for state in ego_future
            ],
        }
    )
    rear_axle = ego_pose.rear_axle
    return {
        "acceleration": 0.0,
        "pose": [rear_axle.x, rear_axle.y, rear_axle.heading],
        "speed": ego_pose.dynamic_car_state.speed,
        "prediction": predictions,
    }


def to_scene_from_states(trajectory: List[EgoState]) -> List[Dict[str, Any]]:
    """
    Convert trajectory into structure
    :param trajectory: input trajectory ot be converted
    :return scene
    """
    return [
        {
            'pose': [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading],
            'speed': state.dynamic_car_state.speed,
            'velocity_2d': [
                state.dynamic_car_state.rear_axle_velocity_2d.x,
                state.dynamic_car_state.rear_axle_velocity_2d.y,
            ],
            'lateral': [0.0, 0.0],
            'acceleration': [
                state.dynamic_car_state.rear_axle_acceleration_2d.x,
                state.dynamic_car_state.rear_axle_acceleration_2d.y,
            ],
            'tire_steering_angle': state.tire_steering_angle,
        }
        for state in trajectory
    ]


def to_scene_from_trajectory(trajectory: AbstractTrajectory) -> List[Dict[str, Any]]:
    """
    Convert trajectory into structure
    :param trajectory: input trajectory ot be converted
    :return scene
    """
    return to_scene_from_states(trajectory.get_sampled_trajectory())


def create_trajectory_structure(trajectory: List[EgoState], color: List[int]) -> Dict[str, Any]:
    """
    Create scene json structure for a trajectory with a color.
    :param trajectory: set of states.
    :param color: color [R, G, B, A]
    :return: scene representing a trajectory
    """
    return {'color': color, 'states': to_scene_from_states(trajectory)}


def to_scene_ego_from_center_pose(state_center: StateSE2) -> Dict[str, Any]:
    """
    Convert to a scene by ego's center pose.
    :param state_center: state of ego's center pose.
    :return scene.
    """
    return {'acceleration': 0.0, 'pose': [state_center.x, state_center.y, state_center.heading], 'speed': 0.0}


def to_scene_agent_type(agent_type: TrackedObjectType) -> str:
    """
    Convert TrackedObjectType to string.
    :param agent_type: TrackedObjectType.
    :return string representing an agent type.
    """
    if agent_type == TrackedObjectType.VEHICLE:
        return 'Vehicle'
    elif agent_type == TrackedObjectType.PEDESTRIAN:
        return 'Pedestrian'
    elif agent_type == TrackedObjectType.BICYCLE:
        return 'Bicycle'
    elif agent_type == TrackedObjectType.GENERIC_OBJECT:
        return 'Generic_object'
    raise ValueError('Unknown input type: {}'.format(str(agent_type)))


def to_scene_box(tracked_object: TrackedObject, track_id: str) -> Dict[str, Any]:
    """
    Convert tracked_object into json representation.
    :param tracked_object: tracked_object representation.
    :param track_id: unique id of a track.
    :return json representation of an agent.
    """
    center_x = tracked_object.center.x
    center_y = tracked_object.center.y
    center_heading = tracked_object.center.heading
    if tracked_object.tracked_object_type in AGENT_TYPES:
        speed = np.hypot(tracked_object.velocity.x, tracked_object.velocity.y)
    else:
        speed = 0
    if track_id is None:
        track_id = 'null'
    scene = {
        'active': True,
        'real': True,
        'speed': speed if not np.isnan(speed) else 0.0,
        'box': {
            'pose': [center_x, center_y, center_heading],
            'size': [tracked_object.box.width, tracked_object.box.length],
        },
        'id': track_id,
        'type': tracked_object.tracked_object_type.fullname,
        'tooltip': f"track_id: {track_id}\ntype: {tracked_object.tracked_object_type.fullname}",
    }
    if tracked_object.tracked_object_type == TrackedObjectType.PEDESTRIAN:
        scene['box']['radius'] = 0.5
    return scene


def to_scene_boxes(tracked_objects: TrackedObjects) -> Dict[str, Any]:
    """
    Convert tracked_objects into a scene.
    :param tracked_objects: List of boxes in global coordinates.
    :return dictionary which should be placed into scene["world"].
    """
    tracked_object_dictionaries = {}
    for track_object_type_name, tracked_object_type in tracked_object_types.items():
        objects = [
            to_scene_box(tracked_object, track_id=tracked_object.track_token)
            for tracked_object in tracked_objects.get_tracked_objects_of_type(tracked_object_type)
        ]
        tracked_object_dictionaries[track_object_type_name] = objects

    return tracked_object_dictionaries


def to_scene_from_ego_and_boxes(
    ego_pose: StateSE2, tracked_objects: TrackedObjects, map_name: str, set_camera: bool = False
) -> Dict[str, Any]:
    """
    Extract scene from ego_pose and boxes.
    :param ego_pose: Ego Position.
    :param tracked_objects: list of actors in global coordinate frame.
    :param map_name: map name.
    :param set_camera: True if we wish to also set camera view.
    :return scene.
    """
    # World
    world = to_scene_boxes(tracked_objects)

    # Extract map name
    map_name_without_suffix = str(pathlib.Path(map_name).stem)

    # Construct Scene
    scene: Dict[str, Any] = {
        'map': {'area': map_name_without_suffix},
        'world': world,
        'ego': to_scene_ego_from_center_pose(ego_pose),
    }

    if set_camera:
        ego_pose = scene['ego']['pose']
        ego_x = ego_pose[0]
        ego_y = ego_pose[1]
        ego_heading = ego_pose[2]
        bearing_rad = np.fmod(ego_heading, np.pi * 2)  # In radians
        if bearing_rad < 0:
            bearing_rad += np.pi * 2
        bearing_rad = 1.75 - (bearing_rad / (np.pi * 2))
        if bearing_rad >= 1:
            bearing_rad -= 1.0
        scene['camera'] = {'pitch': 50, 'scale': 2500000, 'bearing': bearing_rad, 'lookat': [ego_x, ego_y, 0.0]}

    return scene


def _to_scene_agent_prediction(tracked_object: TrackedObject, color: List[int]) -> Dict[str, Any]:
    """
    Extract agent's predicted states from TrackedObject to scene.
    :param tracked_object: tracked_object representation.
    :param color: color [R, G, B, A].
    :return a prediction scene.
    """

    def extract_prediction_state(pose: StateSE2, timestamp: float) -> Dict[str, Any]:
        return {'pose': [pose.x, pose.y, pose.heading], 'polygon': [[pose.x, pose.y]], 'timestamp': timestamp}

    # Convert driven trajectory
    past_states = (
        []
        if not tracked_object.past_trajectory
        else [
            extract_prediction_state(
                waypoint.oriented_box.center, tracked_object.metadata.timestamp_s - waypoint.time_point.time_s
            )
            for waypoint in tracked_object.past_trajectory.waypoints
            if waypoint
        ]
    )

    # Convert future trajectories
    future_states = [
        extract_prediction_state(
            waypoint.oriented_box.center, waypoint.time_point.time_s - mode.waypoints[0].time_point.time_s
        )
        for mode in tracked_object.predictions
        for waypoint in mode.waypoints
        if waypoint
    ]
    return {
        'id': tracked_object.metadata.track_id,
        'color': color,
        'size': [tracked_object.box.width, tracked_object.box.length],
        'states': past_states + future_states,
    }


def to_scene_agent_prediction_from_boxes(tracked_objects: TrackedObjects, color: List[int]) -> List[Dict[str, Any]]:
    """
    Convert predicted observations into prediction dictionary.
    :param tracked_objects: List of tracked_objects in global coordinates.
    :param color: color [R, G, B, A].
    :return scene.
    """
    return [
        _to_scene_agent_prediction(tracked_object, color)
        for tracked_object in tracked_objects
        if tracked_object.predictions is not None
    ]


def to_scene_agent_prediction_from_boxes_separate_color(
    tracked_objects: TrackedObjects, color_vehicles: List[int], color_pedestrians: List[int], color_bikes: List[int]
) -> List[Dict[str, Any]]:
    """
    Convert predicted observations into prediction dictionary.
    :param tracked_objects: List of tracked_objects in global coordinates.
    :param color_vehicles: color [R, G, B, A] for vehicles predictions.
    :param color_pedestrians: color [R, G, B, A] for pedestrians predictions.
    :param color_bikes: color [R, G, B, A] for bikes predictions.
    :return scene.
    """
    predictions = []
    for tracked_object in tracked_objects:

        if tracked_object.predictions is None:
            continue

        if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE:
            color = color_vehicles
        elif tracked_object.tracked_object_type == TrackedObjectType.PEDESTRIAN:
            color = color_pedestrians
        elif tracked_object.tracked_object_type == TrackedObjectType.BICYCLE:
            color = color_bikes
        else:
            color = [0, 0, 0, 255]

        predictions.append(_to_scene_agent_prediction(tracked_object, color))

    return predictions
