"""
Shared tools focused on measuring performance.
"""
import math
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.geometry import quaternion_yaw

# (xmin , ymin, xmax, ymax).
Rectangle = Tuple[float, float, float, float]
# (xcenter, ycenter, width, height, yaw).
TwoDimBox = Tuple[float, float, float, float, float]


def intersection(a: Rectangle, b: Rectangle) -> float:
    """
    Intersection between rectangles.
    :param a: Rectangle 1.
    :param b: Rectangle 2.
    :return: Area of intersection between a and b.
    """
    # find difference in x direction
    dx = min(a[2], b[2]) - max(a[0], b[0])

    # find difference in y direction
    dy = min(a[3], b[3]) - max(a[1], b[1])

    # return intersection
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0


def union(a: Rectangle, b: Rectangle) -> float:
    """
    Union of two rectangles.
    :param a: Rectangle 1.
    :param b: Rectangle 2.
    :return: Area of union between a and b.
    """
    return (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection(a, b)


def birdview_corner_angle_mean_distance(a: TwoDimBox, b: TwoDimBox, period: float) -> float:
    """
    Calculates ad-hoc birdsview distance of two 2-d boxes.
    :param a: 2-d box1.
    :param b: 2-d box2.
    :param period: Periodicity for assessing angle difference.
    :return: Birdview distance.
    """
    # Since distances are in meters and angle in radians, they are mostly on the same scale.
    # E.g 25 degrees difference is 25*np.pi/180 = 0.43.
    # So a simple mean absolute difference actually works fine.
    box_error: npt.NDArray[np.float64] = np.array(a[:4]) - np.array(b[:4])
    yaw_error = angle_diff(a[4], b[4], period)

    avg_abs_error = float(np.mean(np.abs(np.concatenate((box_error, np.array([yaw_error]))))))

    return avg_abs_error


def birdview_corner_angle_mean_distance_box(a: Box3D, b: Box3D, period: float) -> float:
    """
    Calculates ad-hoc birdview distance of two Box3D instances.
    :param a: Box3D 1.
    :param b: Box3D 2.
    :param period: Periodicity for assessing angle difference.
    :return: Birdview distance.
    """
    # Since distances are in meters and angle in radians, they are mostly on the same scale.
    # E.g 25 degrees difference is 25*np.pi/180 = 0.43.
    # So a simple mean absolute difference actually works fine.
    error = 0.0
    error += abs(a.center[0] - b.center[0])  # x error
    error += abs(a.center[1] - b.center[1])  # y error
    error += abs(a.wlh[0] - b.wlh[0])  # width error
    error += abs(a.wlh[1] - b.wlh[1])  # length error

    a_yaw = quaternion_yaw(a.orientation)
    b_yaw = quaternion_yaw(b.orientation)

    error += abs(angle_diff(a_yaw, b_yaw, period))

    return error / 5


def birdview_pseudo_iou_box(a: Box3D, b: Box3D, period: float) -> float:
    """
    Calculates ad-hoc birdview IoU of two Box3D instances.
    :param a: Box3D 1.
    :param b: Box3D 2.
    :param period: Periodicity for assessing angle difference.
    :return: Birdview IoU.
    """
    # IoU will be always between 0 and 1.
    return 1 / (1 + birdview_corner_angle_mean_distance_box(a, b, period))


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity for assessing angle difference.
    :return: Signed smallest between-angle difference in range (-pi, pi).
    """
    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > math.pi:
        diff = diff - (2 * math.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def angle_diff_numpy(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], period: float) -> npt.NDArray[np.float64]:
    """
    Gets the smallest angle difference between 2 arrays of angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity for assessing angle difference.
    :return: Signed smallest between-angle difference in range (-period/2, period/2).
    """
    assert 0 < period <= 2 * np.pi

    # Calculate angle difference, modulo to [0, period].
    diff = (x - y + period / 2) % period - period / 2

    # Shift (pi, 2*pi] to (-pi, 0].
    diff[diff > np.pi] = diff[diff > np.pi] - (2 * np.pi)

    return diff


def hausdorff_distance_box(obsbox: Box3D, gtbox: Box3D) -> float:
    """
    Calculate Hausdorff distance between two 2d-boxes in Box3D class.
    :param obsbox: Observation box.
    :param gtbox: Ground truth box.
    :return: Hausdorff distance.
    """

    def footprint(box: Box3D) -> Polygon:
        """
        Get footprint polygon.
        :param box: (center_x <float>, center_y <float>, width <float>, length <float>, theta <float>).
        :return: <Polygon>. A polygon representation of the 2d box.
        """
        x, y, w, l, head = (box.center[0], box.center[1], box.wlh[0], box.wlh[1], quaternion_yaw(box.orientation))
        rot = np.array([[math.cos(head), -math.sin(head)], [math.sin(head), math.cos(head)]])  # type: ignore
        q0 = np.array([x, y])[:, None]
        q1 = np.array([-w / 2, -l / 2])[:, None]
        q2 = np.array([-w / 2, l / 2])[:, None]
        q3 = np.array([w / 2, l / 2])[:, None]
        q4 = np.array([w / 2, -l / 2])[:, None]
        q1 = np.dot(rot, q1) + q0
        q2 = np.dot(rot, q2) + q0
        q3 = np.dot(rot, q3) + q0
        q4 = np.dot(rot, q4) + q0

        return Polygon(
            [(q1.item(0), q1.item(1)), (q2.item(0), q2.item(1)), (q3.item(0), q3.item(1)), (q4.item(0), q4.item(1))]
        )

    # polygon representations of estbox, gtbox
    obs_poly = footprint(obsbox)
    gt_poly = footprint(gtbox)

    # compute the distance value
    distance = 0.0
    for p in list(gt_poly.exterior.coords):
        new_dist = float(obs_poly.distance(Point(p)))
        if new_dist > distance:
            distance = new_dist

    for p in list(obs_poly.exterior.coords):
        new_dist = float(gt_poly.distance(Point(p)))
        if new_dist > distance:
            distance = new_dist

    return distance


def hausdorff_distance(obsbox: TwoDimBox, gtbox: TwoDimBox) -> float:
    """
    Calculate Hausdorff distance between two 2d-boxes.
    :param obsbox: Observation 2d box.
    :param gtbox: Ground truth 2d box.
    :return: Hausdorff distance.
    """

    def footprint(box: TwoDimBox) -> Polygon:
        """
        Get footprint polygon.
        :param box: Input 2-d box.
        :return: A polygon representation of the 2d box.
        """
        x, y, w, l, head = box
        rot = np.array([[math.cos(head), -math.sin(head)], [math.sin(head), math.cos(head)]])  # type: ignore
        q0 = np.array([x, y])[:, None]
        q1 = np.array([-w / 2, -l / 2])[:, None]
        q2 = np.array([-w / 2, l / 2])[:, None]
        q3 = np.array([w / 2, l / 2])[:, None]
        q4 = np.array([w / 2, -l / 2])[:, None]
        q1 = np.dot(rot, q1) + q0
        q2 = np.dot(rot, q2) + q0
        q3 = np.dot(rot, q3) + q0
        q4 = np.dot(rot, q4) + q0

        return Polygon(
            [(q1.item(0), q1.item(1)), (q2.item(0), q2.item(1)), (q3.item(0), q3.item(1)), (q4.item(0), q4.item(1))]
        )

    # polygon representations of estbox, gtbox
    obs_poly = footprint(obsbox)
    gt_poly = footprint(gtbox)

    # compute the distance value
    distance = 0.0
    for p in list(gt_poly.exterior.coords):
        new_dist = float(obs_poly.distance(Point(p)))
        if new_dist > distance:
            distance = new_dist

    for p in list(obs_poly.exterior.coords):
        new_dist = float(gt_poly.distance(Point(p)))
        if new_dist > distance:
            distance = new_dist

    return distance


def birdview_center_distance_box(a: Box3D, b: Box3D) -> float:
    """
    Calculates the l2 distance between birdsview bounding box centers in Box3D class format.
    :param a: Box3D class.
    :param b: Box3D class.
    :return: Center distance.
    """
    return float(np.sqrt((a.center[0] - b.center[0]) ** 2 + (a.center[1] - b.center[1]) ** 2))


def birdview_center_distance(
    a: Union[Tuple[float, float], TwoDimBox], b: Union[Tuple[float, float], TwoDimBox]
) -> float:
    """
    Calculates the l2 distance between birdsview bounding box centers.
    :param a: (xcenter, ycenter). Also accepts longer representation including width, height, yaw.
    :param b: (xcenter, ycenter). Also accepts longer representation including width, height, yaw.
    :return: Center distance.
    """
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def assign(
    box_list1: List[Any], box_list2: List[Any], dist_fcn: Callable[[Any, Any], float], assign_th: float
) -> List[Tuple[Any, Any]]:
    """
    Runs the hungarian algorithm for bounding box assignments
    :param box_list1: [<BOX_FORMAT>]. List of boxes. BOX_FORMAT much be compatible with dist_fcn inputs.
    :param box_list2: [<BOX_FORMAT>]. List of boxes. BOX_FORMAT much be compatible with dist_fcn inputs.
    :param dist_fcn: <fcn (<BOX_FORMAT>, <BOX_FORMAT>) -> <float>>. Calculates distances between two boxes.
    :param assign_th: Only assign a match if the affinity for a pair is below this threshold.
    :return: [(index_box_list1 <int>, index_box_list2 <int>)]. Pairs of box indices for matches.
    """
    costmatrix = np.zeros((len(box_list1), len(box_list2)))

    for row, gtbox in enumerate(box_list1):
        for col, estbox in enumerate(box_list2):
            costmatrix[row, col] = dist_fcn(gtbox, estbox)

    # Prevent assignment if cost greater than assign_th.
    costmatrix[costmatrix > assign_th] = 1000 * assign_th

    row_ind, col_ind = linear_sum_assignment(costmatrix)
    pairs = zip(row_ind, col_ind)
    pairs_list = [pair for pair in pairs if costmatrix[pair[0], pair[1]] < assign_th]

    return pairs_list


def weighted_harmonic_mean(x: List[float], w: List[float]) -> float:
    """
    Calculate the weighted harmonic mean of x with weights given by w.
    :param x: [<float> * n]. Input data.
    :param w: [<float> * n]. Weights. Needs to be same shape.
    :return: The weighted harmonic mean.
    """
    w = list(map(float, w))

    if any([xi == 0 for xi in x]):
        return 0

    if any([wi <= 0 for wi in w]):
        raise ValueError("w must contain strictly positive entries")

    return float(np.sum(w) / np.sum([wi / xi for xi, wi in zip(x, w)]))


def long_lat_dist_decomposition(
    gt_vector: npt.NDArray[np.float64], est_vector: npt.NDArray[np.float64]
) -> Tuple[float, float]:
    """
    Longitudinal and lateral decomposition of est_vector - gt_vector.
    We define longitudinal direction as the direction of gt_vector. Lateral direction is defined as direction of
    cross product between longitudinal vector and vertical vector (longitudinal x vertical).
    :param gt_vector: <np.float: 2>. 2-dimensional ground truth vector.
    :param est_vector: <np.float: 2>. 2-dimensional ground estimated vector.
    :return: Longitudinal distance and lateral distance.
    """
    assert gt_vector.size == est_vector.size == 2, "Input vector should be 2-dimensional"
    # We specify this condition to handle when gt vector is zero.
    # We consider the error is solely longitudinal in this case.
    if np.all(gt_vector == 0):
        return np.linalg.norm(est_vector), 0  # type: ignore
    unit_long_vector = gt_vector / np.linalg.norm(gt_vector)
    dist_vector: npt.NDArray[np.float64] = est_vector - gt_vector
    long_dist = float(np.dot(unit_long_vector, dist_vector))
    lat_dist = np.linalg.norm(dist_vector - (long_dist * unit_long_vector))

    return long_dist, lat_dist  # type: ignore


def get_euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Gets the straight line distance between two points (generally used for finding the distance between two UTM
    coordinates).
    :param x1: The x-coordinate of the first point.
    :param y1: The y-coordinate of the first point.
    :param x2: The x-coordinate of the second point.
    :param y2: The y-coordinate of the second point.
    :return: The straight line distance between (x1, y1) and (x2, y2).
    """
    dx = x1 - x2
    dy = y1 - y2

    return math.sqrt(dx * dx + dy * dy)
