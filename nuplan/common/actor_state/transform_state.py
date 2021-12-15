from copy import deepcopy
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.database.utils.boxes.box3d import Box3D
from shapely.geometry import Polygon
from shapely.ops import nearest_points


def translate_longitudinally(pose: StateSE2, distance: float) -> Point2D:
    """
    Translate the position component of an SE2 pose longitudinally (along heading direction)
    :param pose: SE2 pose to be translated
    :param distance: [m] distance by which point (x, y, heading) should be translated longitudinally
    :return Point2D translated position
    """
    return Point2D(pose.x + distance * np.cos(pose.heading), pose.y + distance * np.sin(pose.heading))


def translate_longitudinally_se2(pose: StateSE2, distance: float) -> StateSE2:
    """
    Translate an SE2 pose longitudinally (along heading direction)
    :param pose: SE2 pose to be translated
    :param distance: [m] distance by which point (x, y, heading) should be translated longitudinally
    :return translated se2
    """
    x, y = translate_longitudinally(pose, distance)
    pose_updated = deepcopy(pose)  # deep copy to include potential extra data within classes derived from StateSE2
    pose_updated.x = x
    pose_updated.y = y
    return pose_updated


def translate_position(pose: StateSE2, lon: float, lat: float) -> Point2D:
    """
    Translate the position component of an SE2 pose longitudinally and laterally
    :param pose: SE2 pose to be translated
    :param lon: [m] distance by which a point should be translated in longitudinal direction
    :param lat: [m] distance by which a point should be translated in lateral direction
    :return Point2D translated position
    """
    m_pi_2 = np.pi / 2.0
    ego_x = pose.x
    ego_y = pose.y
    ego_heading = pose.heading
    x = ego_x + (lat * np.cos(ego_heading + m_pi_2)) + (lon * np.cos(ego_heading))
    y = ego_y + (lat * np.sin(ego_heading + m_pi_2)) + (lon * np.sin(ego_heading))
    return Point2D(x, y)


def get_front_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the front left corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_position(center_pose, half_length, half_width)


def get_front_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the front right corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_position(center_pose, half_length, -half_width)


def get_rear_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the rear left corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_position(center_pose, -half_length, half_width)


def get_rear_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the rear right corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_position(center_pose, -half_length, -half_width)


def construct_ego_rectangle(ego_state: StateSE2, vehicle: VehicleParameters) -> List[Point2D]:
    """
    Get points for ego's reactangle
    :param ego_state: Ego's x, y (center of the rear axle) and heading
    :param vehicle: parameters pertaining to vehicles
    :return: List of Tuples of all points
    """
    front_left = get_front_left_corner(ego_state, vehicle.front_length, vehicle.half_width)
    front_right = get_front_right_corner(ego_state, vehicle.front_length, vehicle.half_width)
    rear_left = get_rear_left_corner(ego_state, vehicle.rear_length, vehicle.half_width)
    rear_right = get_rear_right_corner(ego_state, vehicle.rear_length, vehicle.half_width)

    return [front_left, front_right, rear_left, rear_right]


def get_ego_polygon(ego_state: StateSE2, vehicle: VehicleParameters) -> Polygon:
    """
    Return Shapely polygon correspoding to Ego
    :param ego_state: x, y (center of rear axle) and heading
    :param vehicle: Parameters of the vehicle
    :return: Shapely polygon for ego
    """
    ego_rectangle = construct_ego_rectangle(ego_state, vehicle)
    # Convert points to a list of [x, y]
    xy_points = [[point.x, point.y] for point in ego_rectangle]
    return Polygon(xy_points)


def get_box_polygon(box: Box3D) -> Polygon:
    """
    Get polygon for a 3DBox
    :param box: Agent's 3D box
    :return: Shapely Polygon for the agent
    """
    agent_box = box.bottom_corners[:2, :].T
    return Polygon(agent_box)


def get_ego_agent_relative_distances(ego_state: StateSE2, box: Box3D,
                                     vehicle_parameters: VehicleParameters) -> Tuple[float, float, float]:
    """
    Get Euclidean, Longitudinal and Lateral distance between ego and agent.
    Euclidean: L2 distance between the closest points of the two boxes
    Longitudinal: L2 distance between centers of two boxes project along ego's heading
        subtracted by half lengths of ego and agent box
    Lateral: L2 distance between centers of two boxes project perpendicular to ego's heading
        subtracted by half widths of ego and agent box
    :param ego_state: ego's x, y (center of the rear axle) and heading
    :param box: Agent's 3D box
    :param vehicle_parameters: Parameters for the ego vehicle
    :return: Dataclass with all distances
    """
    # Construct Ego Polygon
    ego_polygon = get_ego_polygon(ego_state, vehicle_parameters)

    # Construct Agent Polygon
    agent_polygon = get_box_polygon(box)

    # Shapely computes distance between closest points between two polygons
    p1, p2 = nearest_points(agent_polygon, ego_polygon)
    euclidean_distance = p1.distance(p2)

    # Create a vector from ego to agent
    agent_ego_vector = [ego_state.x - box.center[0], ego_state.y - box.center[1]]
    vec_dist = np.linalg.norm(agent_ego_vector)  # type: ignore

    # Get angle the vector from ego to agent makes with x axis
    agent_ego_angle = np.arccos(agent_ego_vector[0] / vec_dist)
    # Get angle ego makes with x axis
    ego_heading = ego_state.heading

    # Get relative angle between ego yaw and agent_ego_vector
    delta_theta = ego_heading - agent_ego_angle

    # Project to get components
    longitudinal_distance, lateral_distance = project_distance_to_lat_lon(vec_dist, delta_theta,
                                                                          vehicle_parameters, box)

    return euclidean_distance, longitudinal_distance, lateral_distance


def project_distance_to_lat_lon(distance: float, angle: float, vehicle: VehicleParameters,
                                box: Box3D) -> Tuple[float, float]:
    """
    Projects distance along lateral and longitudinal directions as per angle
    :param distance: Distance to be projected
    :param angle: Angle used for projection
    :param vehicle: Vehicle parameters
    :param box: Agent box
    :return: Longitudinal and lateral distance
    """
    longitudinal_distance = np.maximum(np.abs(distance * np.cos(angle)
                                              ) - vehicle.half_length - box.length / 2.0, 0)
    lateral_distance = np.maximum(np.abs(distance * np.sin(angle)) -
                                  vehicle.half_width - box.width / 2.0, 0)

    return longitudinal_distance, lateral_distance


def pose_to_matrix(pose: StateSE2) -> npt.NDArray[np.float32]:
    """
    Converts a 2D pose to a 3x3 transformation matrix

    :param pose: 2D pose (x, y, yaw)
    :return: 3x3 transformation matrix
    """
    return np.array(
        [
            [np.cos(pose.heading), -np.sin(pose.heading), pose.x],
            [np.sin(pose.heading), np.cos(pose.heading), pose.y],
            [0, 0, 1],
        ]
    )


def pose_from_matrix(transform: npt.NDArray[np.float32]) -> StateSE2:
    """
    Converts a 3x3 transformation matrix to a 2D pose

    :param transform: 3x3 transformation matrix
    :return: 2D pose (x, y, yaw)
    """
    if transform.shape != (3, 3):
        raise RuntimeError(f'Expected a 3x3 transformation matrix, got {transform.shape}')

    heading = np.arctan2(transform[1, 0], transform[0, 0])

    return StateSE2(transform[0, 2], transform[1, 2], heading)


def convert_absolute_to_relative_poses(absolute_poses: List[StateSE2]) -> List[StateSE2]:
    """
    Converts a list of SE2 poses from absolute to relative coordinates with the first pose being the origin

    :param absolute_poses: list of absolute poses to convert
    :return: list of converted relative poses
    """
    absolute_transforms = np.array([pose_to_matrix(pose) for pose in absolute_poses])
    origin_transform = np.linalg.inv(absolute_transforms[0])  # type: ignore
    relative_transforms = origin_transform @ absolute_transforms
    relative_poses = [pose_from_matrix(transform) for transform in relative_transforms]

    return relative_poses


def convert_relative_to_absolute_poses(origin_pose: StateSE2, relative_poses: List[StateSE2]) -> List[StateSE2]:
    """
    Converts a list of SE2 poses from relative to absolute coordinates using an origin pose.

    :param absolute_poses: list of relative poses to convert
    :return: list of converted absolute poses
    """
    relative_transforms = np.array([pose_to_matrix(pose) for pose in relative_poses])
    origin_transform = pose_to_matrix(origin_pose)
    absolute_transforms: npt.NDArray[np.float32] = origin_transform @ relative_transforms
    relative_poses = [pose_from_matrix(transform) for transform in absolute_transforms]

    return relative_poses
