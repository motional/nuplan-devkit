from functools import lru_cache
from typing import Callable, List, Optional, Set, Tuple, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import ProgressStateSE2, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.geometry.compute import signed_lateral_distance
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from nuplan.planning.simulation.path.utils import calculate_progress


@lru_cache(maxsize=256)
def get_agent_relative_angle(ego_state: StateSE2, agent_state: StateSE2) -> float:
    """
    Get the the relative angle of an agent position to the ego
    :param ego_state: pose of ego
    :param agent_state: pose of an agent
    :return: relative angle in radians.
    """
    agent_vector: npt.NDArray[np.float32] = np.array([agent_state.x - ego_state.x, agent_state.y - ego_state.y])
    ego_vector: npt.NDArray[np.float32] = np.array([np.cos(ego_state.heading), np.sin(ego_state.heading)])
    dot_product = np.dot(ego_vector, agent_vector / np.linalg.norm(agent_vector))
    return float(np.arccos(dot_product))


def is_agent_ahead(ego_state: StateSE2, agent_state: StateSE2, angle_tolerance: float = 30) -> bool:
    """
    Determines if an agent is ahead of the ego
    :param ego_state: ego's pose
    :param agent_state: agent's pose
    :param angle_tolerance: tolerance to consider if agent is ahead, where zero is the heading of the ego [deg]
    :return: true if agent is ahead, false otherwise.
    """
    return bool(get_agent_relative_angle(ego_state, agent_state) < np.deg2rad(angle_tolerance))


def is_agent_behind(ego_state: StateSE2, agent_state: StateSE2, angle_tolerance: float = 150) -> bool:
    """
    Determines if an agent is behind of the ego
    :param ego_state: ego's pose
    :param agent_state: agent's pose
    :param angle_tolerance: tolerance to consider if agent is behind, where zero is the heading of the ego [deg]
    :return: true if agent is behind, false otherwise
    """
    return bool(get_agent_relative_angle(ego_state, agent_state) > np.deg2rad(angle_tolerance))


def get_closest_agent_in_position(
    ego_state: EgoState,
    observations: DetectionsTracks,
    is_in_position: Callable[[StateSE2, StateSE2], bool],
    collided_track_ids: Set[str] = set(),
    lateral_distance_threshold: float = 0.5,
) -> Tuple[Optional[Agent], float]:
    """
    Searches for the closest agent in a specified position
    :param ego_state: ego's state
    :param observations: agents as DetectionTracks
    :param is_in_position: a function to determine the positional relationship to the ego
    :param collided_track_ids: Set of collided track tokens, default {}
    :param lateral_distance_threshold: Agents laterally further away than this threshold are not considered, default 0.5 meters
    :return: the closest agent in the position and the corresponding shortest distance.
    """
    closest_distance = np.inf
    closest_agent = None

    for agent in observations.tracked_objects.get_agents():
        if (
            is_in_position(ego_state.rear_axle, agent.center)
            and agent.track_token not in collided_track_ids
            and abs(signed_lateral_distance(ego_state.rear_axle, agent.box.geometry)) < lateral_distance_threshold
        ):
            distance = abs(ego_state.car_footprint.oriented_box.geometry.distance(agent.box.geometry))
            if distance < closest_distance:
                closest_distance = distance
                closest_agent = agent

    return closest_agent, float(closest_distance)


def ego_path_to_se2(path: List[EgoState]) -> List[StateSE2]:
    """
    Convert a list of EgoState into a list of StateSE2.
    :param path: The path to be converted.
    :return: A list of StateSE2.
    """
    return [state.center for state in path]


def create_path_from_se2(states: List[StateSE2]) -> InterpolatedPath:
    """
    Constructs an InterpolatedPath from a list of StateSE2.
    :param states: Waypoints to construct an InterpolatedPath.
    :return: InterpolatedPath.
    """
    progress_list = calculate_progress(states)

    # Find indices where the progress states are repeated and to be filtered out.
    progress_diff = np.diff(progress_list)
    repeated_states_mask = np.isclose(progress_diff, 0.0)

    progress_states = [
        ProgressStateSE2(progress=progress, x=point.x, y=point.y, heading=point.heading)
        for point, progress, is_repeated in zip(states, progress_list, repeated_states_mask)
        if not is_repeated
    ]
    return InterpolatedPath(progress_states)


def create_path_from_ego_state(states: List[EgoState]) -> InterpolatedPath:
    """
    Constructs an InterpolatedPath from a list of EgoState.
    :param states: waypoints to construct an InterpolatedPath.
    :return InterpolatedPath.
    """
    return create_path_from_se2(ego_path_to_se2(states))


def rotate_vector(
    vector: Tuple[float, float, float],
    theta: float,
    inverse: bool = False,
) -> Tuple[float, float, float]:
    """
    Apply a 2D rotation around the z axis.

    :param vector: the vector to be rotated
    :param theta: the amount to rotate by
    :param inverse: direction of rotation
    :return: the transformed vector.
    """
    assert len(vector) == 3, "vector to be transformed must have length 3"
    rotation_matrix = R.from_rotvec([0, 0, theta])
    if inverse:
        rotation_matrix = rotation_matrix.inv()
    local_vector = rotation_matrix.apply(vector)
    return cast(Tuple[float, float, float], local_vector.tolist())


def transform_vector_global_to_local_frame(
    vector: Tuple[float, float, float], theta: float
) -> Tuple[float, float, float]:
    """
    Transform a vector from global frame to local frame.

    :param vector: the vector to be rotated
    :param theta: the amount to rotate by
    :return: the transformed vector.
    """
    return rotate_vector(vector, theta)


def transform_vector_local_to_global_frame(
    vector: Tuple[float, float, float], theta: float
) -> Tuple[float, float, float]:
    """
    Transform a vector from local frame to global frame.

    :param vector: the vector to be rotated
    :param theta: the amount to rotate by
    :return: the transformed vector.
    """
    return rotate_vector(vector, theta, inverse=True)


def path_to_linestring(path: List[StateSE2]) -> LineString:
    """
    Converts a List of StateSE2 into a LineString
    :param path: path to be converted
    :return: LineString.
    """
    return LineString([(point.x, point.y) for point in path])


def ego_path_to_linestring(path: List[EgoState]) -> LineString:
    """
    Converts a List of EgoState into a LineString
    :param path: path to be converted
    :return: LineString.
    """
    return path_to_linestring(ego_path_to_se2(path))


def is_track_stopped(tracked_object: TrackedObject, stopped_speed_threshhold: float = 5e-02) -> bool:
    """
    Evaluates if a tracked object is stopped
    :param tracked_object: tracked_object representation
    :param stopped_speed_threshhold: Threshhold for 0 speed due to noise
    :return: True if track is stopped else False.
    """
    return (
        True
        if not isinstance(tracked_object, Agent)
        else bool(tracked_object.velocity.magnitude() <= stopped_speed_threshhold)
    )
