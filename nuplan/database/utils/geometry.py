import math

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculates the yaw angle from a quaternion.
    Follow convention: R = Rz(yaw)Ry(pitch)Px(roll)
    Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """
    a = 2.0 * (q[0] * q[3] + q[1] * q[2])
    b = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)

    return math.atan2(a, b)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """
    Calculate the quaternion from a yaw angle.
    :param yaw: yaw angle
    :return: Quaternion
    """
    return Quaternion(axis=(0, 0, 1), radians=yaw)


def transform_matrix(
    translation: npt.NDArray[np.float64] = np.array([0, 0, 0]),
    rotation: Quaternion = Quaternion([1, 0, 0, 0]),
    inverse: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Converts pose to transform matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w, ri, rj, rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def view_points(
    points: npt.NDArray[np.float64], view: npt.NDArray[np.float64], normalize: bool
) -> npt.NDArray[np.float64]:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def minimum_bounding_rectangle(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Finds the smallest bounding rectangle for a set of points in two dimensional space.
    Returns a set of points (in clockwise order) representing the corners of the bounding box.

    Algorithm high level idea:
        One edge of the minimum bounding rectangle for a set of points will be the same as one of the edges of the
        convex hull of those points.

    Algorithm:
     1. Create a convex hull (https://en.wikipedia.org/wiki/Convex_hull) of the input points.
     2. Calculate the angles that all the edges of the convex hull make with the x-axis. Assume that there are N unique
        angles calculated in this step.
     3. Create rotation matrices for all the N unique angles computed in step 2.
     4. Create N set of convex hull points by rotating the original convex hull points using all the N rotation matrices
        computed in the last step.
     5. For each of the N set of convex hull points computed in the last step, calculate the bounding rectangle by
        calculating (min_x, max_x, min_y, max_y).
     6. For the N bounding rectangles computed in the last step, find the rectangle with the minimum area. This will
        give the minimum bounding rectangle for our rotated set of convex hull points (see Step 4).
     7. Undo the rotation of the convex hull by multiplying the points with the inverse of the rotation matrix. And
        remember that the inverse of a rotation matrix is equal to the transpose of the rotation matrix. The returned
        points are in a clockwise order.

    To visualize what this function does, you can use the following snippet:

    for n in range(10):
        points = np.random.rand(8,2)
        plt.scatter(points[:,0], points[:,1])
        bbox = minimum_bounding_rectangle(points)
        plt.fill(bbox[:,0], bbox[:,1], alpha=0.2)
        plt.axis('equal')
        plt.show()

    :param points: <nbr_points, 2>. A nx2 matrix of coordinates where n >= 3.
    :return: A 4x2 matrix of coordinates of the minimum bounding rectangle (in clockwise order).
    """
    assert points.ndim == 2, "Points ndim should be 2."
    assert points.shape[1] == 2, "Points shape: n x 2 where n>= 3."
    assert points.shape[0] >= 3, "Points shape: n x 2 where n>= 3."

    pi2 = np.pi / 2.0

    # Get the convex hull for the points.
    hull_points = points[ConvexHull(points).vertices]

    # Calculate the angles that the edges of the convex hull make with the x-axis.
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # Find rotation matrices for all the unique angles.
    rotations = np.vstack(
        [np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]
    ).T  # type: ignore

    rotations = rotations.reshape((-1, 2, 2))

    # Apply rotations to the hull.
    rot_points = np.dot(rotations, hull_points.T)

    # Find the bounding rectangle for each set of points.
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # Find the bounding rectangle with the minimum area.
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # Find the coordinates and the rotation matrix of the minimum bounding rectangle.
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    pts_clockwise_order = np.zeros((4, 2))
    pts_clockwise_order[0] = np.dot([x1, y2], r)
    pts_clockwise_order[1] = np.dot([x2, y2], r)
    pts_clockwise_order[2] = np.dot([x2, y1], r)
    pts_clockwise_order[3] = np.dot([x1, y1], r)

    return pts_clockwise_order
