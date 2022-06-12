from typing import List, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from shapely.geometry import Polygon

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.transform import translate_laterally, translate_longitudinally


def lateral_distance(reference: StateSE2, other: Point2D) -> float:
    """
    Lateral distance from a point to a reference pose
    :param reference: the reference pose
    :param other: the query point
    :return: the lateral distance
    """
    return float(
        -np.sin(reference.heading) * (other.x - reference.x) + np.cos(reference.heading) * (other.y - reference.y)
    )


def longitudinal_distance(reference: StateSE2, other: Point2D) -> float:
    """
    Longitudinal distance from a point to a reference pose
    :param reference: the reference pose
    :param other: the query point
    :return: the longitudinal distance
    """
    return float(
        np.cos(reference.heading) * (other.x - reference.x) + np.sin(reference.heading) * (other.y - reference.y)
    )


def signed_lateral_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    Computes the minimal lateral distance of ego from another polygon
    :param ego_state: the state of ego
    :param other: the query polygon
    :return: the signed lateral distance
    """
    ego_half_width = get_pacifica_parameters().half_width
    ego_left = translate_laterally(ego_state, ego_half_width)
    ego_right = translate_laterally(ego_state, -ego_half_width)

    vertices = list(zip(*other.exterior.coords.xy))
    distance_left = max(min(lateral_distance(ego_left, Point2D(*vertex)) for vertex in vertices), 0)
    distance_right = max(min(-lateral_distance(ego_right, Point2D(*vertex)) for vertex in vertices), 0)
    return distance_left if distance_left > distance_right else -distance_right


def signed_longitudinal_distance(ego_state: StateSE2, other: Polygon) -> float:
    """
    Computes the minimal longitudinal distance of ego from another polygon
    :param ego_state: the state of ego
    :param other: the query polygon
    :return: the signed lateral distance
    """
    ego_half_length = get_pacifica_parameters().half_length
    ego_front = translate_longitudinally(ego_state, ego_half_length)
    ego_back = translate_longitudinally(ego_state, -ego_half_length)

    vertices = list(zip(*other.exterior.coords.xy))
    distance_front = max(min(longitudinal_distance(ego_front, Point2D(*vertex)) for vertex in vertices), 0)
    distance_back = max(min(-longitudinal_distance(ego_back, Point2D(*vertex)) for vertex in vertices), 0)
    return distance_front if distance_front > distance_back else -distance_back


def compute_distance(lhs: StateSE2, rhs: StateSE2) -> float:
    """
    Compute the euclidean distance between two points
    :param lhs: first point
    :param rhs: second point
    :return distance between two points
    """
    return float(np.hypot(lhs.x - rhs.x, lhs.y - rhs.y))


def compute_lateral_displacements(poses: List[StateSE2]) -> List[float]:
    """
    Computes the lateral displacements (y_t - y_t-1) from a list of poses

    :param poses: list of N poses to compute displacements from
    :return: list of N-1 lateral displacements
    """
    return [poses[idx].y - poses[idx - 1].y for idx in range(1, len(poses))]


def principal_value(
    angle: Union[float, int, npt.NDArray[np.float64]], min_: float = -np.pi
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Wrap heading angle in to specified domain (multiples of 2 pi alias),
    ensuring that the angle is between min_ and min_ + 2 pi. This function raises an error if the angle is infinite
    :param angle: rad
    :param min_: minimum domain for angle (rad)
    :return angle wrapped to [min_, min_ + 2 pi).
    """
    assert np.all(np.isfinite(angle)), "angle is not finite"

    lhs = (angle - min_) % (2 * np.pi) + min_

    return lhs


class AngularInterpolator:
    """Creates an angular linear interpolator."""

    def __init__(self, states: npt.NDArray[np.float64], angular_states: npt.NDArray[np.float64]):
        """
        :param states: x values for interpolation
        :param angular_states: y values for interpolation
        """
        _angular_states = np.unwrap(angular_states, axis=0)

        self.interpolator = interp1d(states, _angular_states, axis=0)

    def interpolate(self, sampled_state: float) -> npt.NDArray[np.float64]:
        """
        Interpolates a single state
        :param sampled_state: The state at which to perform interpolation
        :return: The value of the state interpolating linearly at the given state
        """
        return principal_value(self.interpolator(sampled_state))  # type: ignore
