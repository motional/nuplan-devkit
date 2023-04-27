import logging
from typing import Any, Dict, List

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.geometry.transform import translate_longitudinally
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.planning.utils.serialization.scene_simple_trajectory import SceneSimpleTrajectory

logger = logging.getLogger(__file__)


def to_state_from_scene(scene: Dict[str, Any]) -> StateSE2:
    """
    Extract state se2 from pose.
    :param scene: position from scene.
    :return StateSE2.
    """
    return StateSE2(x=scene['pose'][0], y=scene['pose'][1], heading=scene['pose'][2])


def to_ego_center_from_scene(scene: Dict[str, Any], vehicle: VehicleParameters) -> StateSE2:
    """
    :param scene: from scene['ego'].
    :param vehicle: vehicle parameters.
    :return the extracted State in the center of ego's bounding box.
    """
    ego_pose = scene['pose']

    ego_x = ego_pose[0]
    ego_y = ego_pose[1]
    ego_heading = ego_pose[2]
    distance = vehicle.rear_axle_to_center
    return translate_longitudinally(StateSE2(ego_x, ego_y, ego_heading), distance)


def to_agent_state_from_scene(
    scene: Dict[str, Any], vehicle: VehicleParameters, time_us: int = 10, to_cog: bool = True
) -> EgoState:
    """
    Extract agent state from scene.
    :param scene: from json.
    :param vehicle: parameters.
    :param time_us: [us] initial time.
    :param to_cog: If true, xy will be translated to the COG of the car. Otherwise xy is assumed to be at rear axle.
    :return EgoState.
    """
    if to_cog:
        ego_state2d = to_ego_center_from_scene(scene, vehicle)
    else:
        ego_pose = scene['pose']
        ego_state2d = StateSE2(ego_pose[0], ego_pose[1], ego_pose[2])

    # x-directions: longitudinal, y-directions: lateral.
    if 'velocity_x' not in scene or 'velocity_y' not in scene:
        velocity_x = scene['speed']
        velocity_y = 0.0
    else:
        velocity_x = scene['velocity_x']
        velocity_y = scene['velocity_y']

    if 'acceleration_x' not in scene or 'acceleration_y' not in scene:
        acceleration_x = scene['acceleration']
        acceleration_y = 0.0
    else:
        acceleration_x = scene['acceleration_x']
        acceleration_y = scene['acceleration_y']

    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(x=ego_state2d.x, y=ego_state2d.y, heading=ego_state2d.heading),
        time_point=TimePoint(time_us),
        rear_axle_velocity_2d=StateVector2D(x=velocity_x, y=velocity_y),
        rear_axle_acceleration_2d=StateVector2D(x=acceleration_x, y=acceleration_y),
        tire_steering_angle=0,
        vehicle_parameters=get_pacifica_parameters(),
    )


def from_scene_tracked_object(scene: Dict[str, Any], object_type: TrackedObjectType) -> TrackedObject:
    """
    Convert scene to a TrackedObject.
    :param scene: scene of an agent.
    :param object_type: type of the resulting object.
    :return Agent extracted from a scene.
    """
    token = scene['id']
    box = scene['box']
    pose = box['pose']
    size = box['size'] if 'size' in box.keys() else [0.5, 0.5]
    default_height = 1.5
    box = OrientedBox(StateSE2(*pose), width=size[0], length=size[1], height=default_height)
    if object_type in AGENT_TYPES:
        return Agent(
            metadata=SceneObjectMetadata(token=str(token), track_token=str(token), track_id=token, timestamp_us=0),
            tracked_object_type=object_type,
            oriented_box=box,
            velocity=StateVector2D(scene['speed'], 0),
        )
    else:
        return StaticObject(
            metadata=SceneObjectMetadata(token=str(token), track_token=str(token), track_id=token, timestamp_us=0),
            tracked_object_type=object_type,
            oriented_box=box,
        )


def from_scene_to_tracked_objects(scene: Dict[str, Any]) -> TrackedObjects:
    """
    Convert scene["world"] into boxes
    :param scene: scene["world"] coming from json
    :return List of boxes representing all agents
    """
    if "world" in scene.keys():
        raise ValueError("You need to pass only the 'world' field of scene, not the whole dict!")
    tracked_objects: List[TrackedObject] = []
    scene_labels_map = {
        'vehicles': TrackedObjectType.VEHICLE,
        'bicycles': TrackedObjectType.BICYCLE,
        'pedestrians': TrackedObjectType.PEDESTRIAN,
    }
    for label, object_type in scene_labels_map.items():
        if label in scene:
            tracked_objects.extend(
                [from_scene_tracked_object(scene_object, object_type) for scene_object in scene[label]]
            )

    return TrackedObjects(tracked_objects)


def from_scene_to_tracked_objects_with_scene_predictions(scene: Dict[str, Any]) -> TrackedObjects:
    """
    Creates tracked objects, loading the predictions directly from the scene json.
    :param scene: The input scene loaded from the json file.
    :return: Tracked objects from the scene, with predictions loaded from the scene json.
    """
    tracked_objects = from_scene_to_tracked_objects(scene["world"])
    tracked_objects_map: Dict[str, TrackedObject] = {track.token: track for track in tracked_objects}

    for prediction in scene["prediction"]:
        prediction_id = str(prediction["id"])
        if prediction_id not in tracked_objects_map:
            logger.warning("Json scene file contains prediction not assigned to any track: %s.", prediction_id)
            continue

        box = tracked_objects_map[prediction_id].box
        current_state = {
            "timestamp": tracked_objects_map[prediction_id].metadata.timestamp_s,
            "pose": list(tracked_objects_map[prediction_id].center),
        }

        tracked_objects_map[prediction_id].predictions = [
            PredictedTrajectory(
                probability=mode["probability"],
                waypoints=SceneSimpleTrajectory(
                    _validate_an_unite_predictions(current_state, mode['states']),
                    width=box.width,
                    length=box.length,
                    height=box.height,
                ).get_sampled_trajectory(),
            )
            for mode in prediction["modes"]
        ]

    return tracked_objects


def _validate_an_unite_predictions(
    current_state: Dict[str, Any], future_states: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Checks that the states are in a temporally consistent order, then builds the prediction with current_state
    and the rest of the future states.
    :param current_state: The current state of the tracked object.
    :param future_states: Future states of prediction.
    :return: Prediction containing the current state as first element.
    """
    if current_state["timestamp"] >= future_states[0]["timestamp"]:
        raise ValueError("Timestamp of first state of future states must be larger than the track's timestamp.")
    for prev_state, state in zip(future_states, future_states[1:]):
        if prev_state["timestamp"] >= state["timestamp"]:
            raise ValueError("The predictions states must be in strictly increasing temporal order!")
    return [current_state] + future_states


def from_scene_to_tracked_objects_with_predictions(
    scene: Dict[str, Any], predictions: List[Dict[str, Any]]
) -> TrackedObjects:
    """
    Creates tracked objects, adding prediction from the given parameter.
    :param scene: The input scene loaded from the json file.
    :param predictions: Predictions for the tracked objects in the scene.
    :return: Tracked objects from the scene, with predictions loaded from the input.
    """
    tracked_objects = from_scene_to_tracked_objects(scene)

    for tracked_object in tracked_objects:
        for prediction in predictions:
            if str(prediction['id']) == str(tracked_object.token):
                box = tracked_object.box
                tracked_object.predictions = [
                    PredictedTrajectory(
                        probability=1.0,
                        waypoints=SceneSimpleTrajectory(
                            prediction['states'], width=box.width, length=box.length, height=box.height
                        ).get_sampled_trajectory(),
                    )
                ]
                del prediction
                break

    return tracked_objects
