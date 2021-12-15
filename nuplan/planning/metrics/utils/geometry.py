import numpy as np
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.transform_state import translate_position
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from shapely.geometry import Polygon


def lateral_distance(reference: StateSE2, other: Point2D) -> float:
    """
    Lateral distance from a point to a reference pose
    :param reference: the reference pose
    :param other: the query point
    :return: the lateral distance
    """
    return float(
        -np.sin(reference.heading) * (other.x - reference.x) + np.cos(reference.heading) * (other.y - reference.y))


def longitudinal_distance(reference: StateSE2, other: Point2D) -> float:
    """
    Longitudinal distance from a point to a reference pose
    :param reference: the reference pose
    :param other: the query point
    :return: the longitudinal distance
    """
    return float(
        np.cos(reference.heading) * (other.x - reference.x) + np.sin(reference.heading) * (other.y - reference.y))


def signed_lateral_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    Computes the minimal lateral distance of ego from another polygon
    :param ego_state: the state of ego
    :param other: the query polygon
    :return: the signed lateral distance
    """
    ego_half_width = get_pacifica_parameters().half_width
    ego_left = StateSE2(*translate_position(ego_state, 0.0, ego_half_width), ego_state.heading)
    ego_right = StateSE2(*translate_position(ego_state, 0.0, -ego_half_width), ego_state.heading)

    vertices = list(zip(*other.exterior.coords.xy))
    distance_left = max(min([lateral_distance(ego_left, Point2D(*vertex)) for vertex in vertices]), 0)
    distance_right = max(min([-lateral_distance(ego_right, Point2D(*vertex)) for vertex in vertices]), 0)
    return distance_left if distance_left > distance_right else -distance_right


def signed_longitudinal_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    Computes the minimal longitudinal distance of ego from another polygon
    :param ego_state: the state of ego
    :param other: the query polygon
    :return: the signed lateral distance
    """
    ego_half_length = get_pacifica_parameters().half_length
    ego_front = StateSE2(*translate_position(ego_state, ego_half_length, 0.0), ego_state.heading)
    ego_back = StateSE2(*translate_position(ego_state, -ego_half_length, 0.0), ego_state.heading)

    vertices = list(zip(*other.exterior.coords.xy))
    distance_front = max(min([longitudinal_distance(ego_front, Point2D(*vertex)) for vertex in vertices]), 0)
    distance_back = max(min([-longitudinal_distance(ego_back, Point2D(*vertex)) for vertex in vertices]), 0)
    return distance_front if distance_front > distance_back else -distance_back
