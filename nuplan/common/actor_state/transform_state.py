from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.transform import translate_longitudinally_and_laterally


def get_front_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the front left corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, half_length, half_width).point


def get_front_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the front right corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, half_length, -half_width).point


def get_rear_left_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the rear left corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, -half_length, half_width).point


def get_rear_right_corner(center_pose: StateSE2, half_length: float, half_width: float) -> Point2D:
    """
    Compute the position of the rear right corner given a center pose and dimensions
    :param center_pose: SE2 pose of the vehicle center to be translated a vehicle corner
    :param half_length: [m] half length of a vehicle's footprint
    :param half_width: [m] half width of a vehicle's footprint
    :return Point2D translated coordinates
    """
    return translate_longitudinally_and_laterally(center_pose, -half_length, -half_width).point
