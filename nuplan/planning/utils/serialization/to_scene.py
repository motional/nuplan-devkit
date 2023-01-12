import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.ego_temporal_state import EgoTemporalState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.utils.color import Color, ColorType
from nuplan.planning.utils.serialization.scene import EgoScene, GoalScene, Trajectory, TrajectoryState

tracked_object_types = {
    'vehicles': TrackedObjectType.VEHICLE,
    'pedestrians': TrackedObjectType.PEDESTRIAN,
    'bicycles': TrackedObjectType.BICYCLE,
    'genericobjects': TrackedObjectType.GENERIC_OBJECT,
    'traffic_cone': TrackedObjectType.TRAFFIC_CONE,
    'barrier': TrackedObjectType.BARRIER,
    'czone_sign': TrackedObjectType.CZONE_SIGN,
}


def to_scene_waypoint(waypoint: Waypoint, time_offset: Optional[float] = None) -> Dict[str, Any]:
    """
    Convert waypoint to scene object that can be visualized as predictions, and offset timestamp if desired
    :param waypoint: to be converted
    :param time_offset: if None, no offset will be done, otherwise offset time stamp by this number
    :return: serialized scene
    """
    return {
        "pose": [waypoint.center.x, waypoint.center.y, waypoint.center.heading],
        "timestamp": waypoint.time_point.time_s + time_offset if time_offset else 0.0,
    }


def to_scene_ego_from_ego_state(ego_pose: Union[EgoState, EgoTemporalState]) -> EgoScene:
    """
    :param ego_pose: temporal state trajectory
    :return serialized scene
    """
    ego_temporal_state = EgoTemporalState(ego_pose) if isinstance(ego_pose, EgoState) else ego_pose
    current_state = ego_temporal_state.ego_current_state

    # Future and past prediction
    future = (
        [
            to_scene_waypoint(state, -current_state.time_point.time_s)
            for prediction in ego_temporal_state.predictions
            for state in prediction.valid_waypoints
        ]
        if ego_temporal_state.predictions
        else []
    )
    past = (
        [
            to_scene_waypoint(state, -current_state.time_point.time_s)
            for state in ego_temporal_state.past_trajectory.valid_waypoints
        ]
        if ego_temporal_state.past_trajectory
        else []
    )

    predictions = {
        "color": Color(red=1, green=0, blue=0, alpha=1, serialize_to=ColorType.FLOAT).to_list(),
        "states": past + future,
    }

    rear_axle = current_state.rear_axle
    return EgoScene(
        acceleration=0.0,
        pose=rear_axle,
        speed=current_state.dynamic_car_state.speed,
        prediction=predictions,
    )


def to_scene_trajectory_state_from_ego_state(ego_state: EgoState) -> TrajectoryState:
    """
    Convert ego state into scene structure for states in a trajectory.
    :param ego_state: ego state.
    :return: state in scene format.
    """
    return TrajectoryState(
        pose=ego_state.rear_axle,
        speed=ego_state.dynamic_car_state.speed,
        velocity_2d=[
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
        ],
        lateral=[0.0, 0.0],
        acceleration=[
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
        ],
        tire_steering_angle=ego_state.tire_steering_angle,
    )


def to_scene_trajectory_from_list_ego_state(trajectory: List[EgoState], color: Color) -> Trajectory:
    """
    Convert list of ego states and a color into a scene structure for a trajectory.
    :param trajectory: a list of states.
    :param color: color [R, G, B, A].
    :return: Trajectory in scene format.
    """
    trajectory_states = [to_scene_trajectory_state_from_ego_state(state) for state in trajectory]
    return Trajectory(color=color, states=trajectory_states)


def to_scene_trajectory_state_from_waypoint(waypoint: Waypoint) -> TrajectoryState:
    """
    Convert ego state into scene structure for states in a trajectory.
    :param waypoint: waypoint in a trajectory.
    :return: state in scene format.
    """
    return TrajectoryState(
        pose=waypoint.center,
        speed=waypoint.velocity.magnitude(),
        velocity_2d=[waypoint.velocity.x, waypoint.velocity.y] if waypoint.velocity else [0, 0],
        lateral=[0.0, 0.0],
    )


def to_scene_trajectory_from_list_waypoint(trajectory: List[Waypoint], color: Color) -> Trajectory:
    """
    Convert list of waypoints and a color into a scene structure for a trajectory.
    :param trajectory: a list of states.
    :param color: color [R, G, B, A].
    :return: Trajectory in scene format.
    """
    trajectory_states = [to_scene_trajectory_state_from_waypoint(state) for state in trajectory]
    return Trajectory(color=color, states=trajectory_states)


def to_scene_goal_from_state(state: StateSE2) -> GoalScene:
    """
    Convert car footprint to scene structure for ego.
    :param car_footprint: CarFootprint of ego.
    :return Ego in scene format.
    """
    return GoalScene(pose=state)


def to_scene_ego_from_car_footprint(car_footprint: CarFootprint) -> EgoScene:
    """
    Convert car footprint to scene structure for ego.
    :param car_footprint: CarFootprint of ego.
    :return Ego in scene format.
    """
    return EgoScene(acceleration=0.0, pose=car_footprint.rear_axle, speed=0.0)


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
        'tooltip': f"avtest_track_id: {track_id}\n"
        f"track_token: {tracked_object.metadata.track_token}\n"
        f"token: {tracked_object.metadata.token}\n"
        f"category_name: {tracked_object.metadata.category_name}\n"
        f"track_id: {tracked_object.metadata.track_id}\n"
        f"type: {tracked_object.tracked_object_type.fullname}\n"
        f"velocity: {tracked_object.velocity}",
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
        'ego': dict(
            to_scene_ego_from_car_footprint(CarFootprint.build_from_center(ego_pose, get_pacifica_parameters()))
        ),
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


def _to_scene_agent_prediction(tracked_object: TrackedObject, color: Color) -> Dict[str, Any]:
    """
    Extract agent's predicted states from TrackedObject to scene.
    :param tracked_object: tracked_object representation.
    :param color: color [R, G, B, A].
    :return a prediction scene.
    """

    def extract_prediction_state(pose: StateSE2, time_delta: float, speed: float) -> Dict[str, Any]:
        """
        Extract the representation of prediction state for scene.
        :param pose: Track pose.
        :param time_delta: Time difference from initial timestamp.
        :param speed: Speed of track.
        :return: Scene-like dict containing prediction state.
        """
        return {
            'pose': [pose.x, pose.y, pose.heading],
            'polygon': [[pose.x, pose.y]],
            'timestamp': time_delta,
            'speed': speed,
        }

    # Convert driven trajectory
    past_states = (
        []
        if not tracked_object.past_trajectory
        else [
            extract_prediction_state(
                waypoint.oriented_box.center,
                tracked_object.metadata.timestamp_s - waypoint.time_point.time_s,
                waypoint.velocity.magnitude() if waypoint.velocity is not None else 0,
            )
            for waypoint in tracked_object.past_trajectory.waypoints
            if waypoint
        ]
    )

    # Convert future trajectories
    future_states = [
        extract_prediction_state(
            waypoint.oriented_box.center,
            waypoint.time_point.time_s - mode.waypoints[0].time_point.time_s,
            waypoint.velocity.magnitude() if waypoint.velocity is not None else 0,
        )
        for mode in tracked_object.predictions
        for waypoint in mode.waypoints
        if waypoint
    ]
    return {
        'id': tracked_object.metadata.track_id,
        'color': color.to_list(),
        'size': [tracked_object.box.width, tracked_object.box.length],
        'states': past_states + future_states,
    }


def to_scene_agent_prediction_from_boxes(tracked_objects: TrackedObjects, color: Color) -> List[Dict[str, Any]]:
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
    tracked_objects: TrackedObjects, color_vehicles: Color, color_pedestrians: Color, color_bikes: Color
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
            color = Color(0, 0, 0, 1, ColorType.FLOAT)

        predictions.append(_to_scene_agent_prediction(tracked_object, color))

    return predictions
