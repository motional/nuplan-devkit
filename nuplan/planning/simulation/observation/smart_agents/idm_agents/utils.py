from typing import Callable, List, Optional, Tuple, cast

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.geometry import quaternion_yaw, yaw_to_quaternion
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from nuplan.planning.simulation.path.utils import convert_se2_path_to_progress_path
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString, Polygon


def get_agent_relative_angle(ego_state: StateSE2, agent_state: StateSE2) -> float:
    """
    Get the the relative angle of an agent position to the ego

    :param ego_state: pose of ego
    :param agent_state: pose of an agent
    :return: relative angle in radians
    """
    agent_vector = np.array([agent_state.x - ego_state.x, agent_state.y - ego_state.y])
    ego_vector = np.array([np.cos(ego_state.heading), np.sin(ego_state.heading)])
    dot_product = np.dot(ego_vector, agent_vector / np.linalg.norm(agent_vector))  # type: ignore
    return float(np.arccos(dot_product))


def is_agent_ahead(ego_state: StateSE2, agent_state: StateSE2, angle_tolerance: float = 20) -> bool:
    """
    Determines if an agent is ahead of the ego

    :param ego_state: ego's pose
    :param agent_state: agent's pose
    :param angle_tolerance: tolerance to consider if agent is behind, where zero is the heading of the ego [deg]
    :return: true if agent is ahead, false otherwise
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


def get_closest_agent_in_position(ego_state: EgoState,
                                  observations: Detections,
                                  is_in_position: Callable[[StateSE2, StateSE2], float]) \
        -> Tuple[Optional[Box3D], float]:
    """
    Searches for the closest agent in a specified position

    :param ego_state: ego's state
    :param observations: agents as Detections
    :param is_in_position: a function to determine the positional relationship to the ego
    :return: the closest agent in the position and the corresponding shortest distance
    """

    closest_distance = np.inf
    closest_agent = None

    for agent in observations.boxes:
        agent_state = StateSE2(x=agent.center[0], y=agent.center[1], heading=quaternion_yaw(agent.orientation))
        if is_in_position(ego_state.rear_axle, agent_state):
            agent_poly = box3d_to_polygon(agent)
            distance = abs(ego_state.car_footprint.oriented_box.geometry.distance(agent_poly))
            if distance < closest_distance:
                closest_distance = distance
                closest_agent = agent

    return closest_agent, float(closest_distance)


def convert_box3d_to_se2(agent_box: Box3D) -> StateSE2:
    """
    Converts Box3D to StateSE2

    :param agent_box: agent as Box3D
    :return: agent as StateSE2
    """
    return StateSE2(x=agent_box.center[0], y=agent_box.center[1], heading=agent_box.yaw)


def create_path_from_se2(states: List[StateSE2]) -> InterpolatedPath:
    """
    Constructs an InterpolatedPath from a list of StateSE2

    :param states: waypoints to construct an InterpolatedPath
    :return: InterpolatedPath
    """
    return InterpolatedPath(convert_se2_path_to_progress_path(states))


def rotate_vector(vector: Tuple[float, float, float], theta: float, inverse: bool = False) \
        -> Tuple[float, float, float]:
    """
    Apply a 2D rotation around the z axis

    :param vector: the vector to be rotated
    :param theta: the amount to rotate by
    :param inverse: direction of rotation
    :return: the transformed vector
    """

    assert len(vector) == 3, "vector to be transformed must have length 3"
    rotation_matrix = R.from_rotvec([0, 0, theta])
    if inverse:
        rotation_matrix = rotation_matrix.inv()
    local_vector = rotation_matrix.apply(vector)
    return cast(Tuple[float, float, float], local_vector.tolist())


def transform_vector_global_to_local_frame(vector: Tuple[float, float, float], theta: float) \
        -> Tuple[float, float, float]:
    """
    transform a vector from global frame to local frame

    :param vector: the vector to be rotated
    :param theta: the amount to rotate by
    :return: the transformed vector
    """
    return rotate_vector(vector, theta)


def transform_vector_local_to_global_frame(vector: Tuple[float, float, float], theta: float) \
        -> Tuple[float, float, float]:
    """
    transform a vector from local frame to global frame

    :param vector: the vector to be rotated
    :param theta: the amount to rotate by
    :return: the transformed vector
    """
    return rotate_vector(vector, theta, inverse=True)


def ego_state_to_box_3d(agent_state: EgoState) -> Box3D:
    """
    Converts EgoState to Box3D
    :param agent_state: agent state to be converted
    :return: agent state as Box3D
    """
    pacifica = get_pacifica_parameters()

    return Box3D(center=(agent_state.center.x,
                         agent_state.center.y, 0),
                 size=(pacifica.width, pacifica.length, 2),
                 orientation=yaw_to_quaternion(agent_state.center.heading),
                 label=1,
                 velocity=(agent_state.dynamic_car_state.center_velocity_2d.x,
                           agent_state.dynamic_car_state.center_velocity_2d.y, 0))


def box3d_to_polygon(box: Box3D) -> Polygon:
    """
    Converts Box3D to a shapely Polygon
    :param box: Box3D to be converted
    :return: Polygon
    """
    agent_box = box.bottom_corners[:2, :].T
    return Polygon(agent_box)


def path_to_linestring(path: List[StateSE2]) -> LineString:
    """
    Converts a List of StateSE2 into a LineString
    :param path: path to be converted
    :return: LineString
    """
    return LineString([(point.x, point.y) for point in path])
