import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import Point2D, StateSE2


def rotate_2d(point: Point2D, rotation_matrix: npt.NDArray[np.float64]) -> Point2D:
    """
    Rotate 2D point with a 2d rotation matrix
    :param point: to be rotated
    :param rotation_matrix: [[R11, R12], [R21, R22]]
    :return: rotated point
    """
    assert rotation_matrix.shape == (2, 2)
    rotated_point = np.array([point.x, point.y]) @ rotation_matrix
    return Point2D(rotated_point[0], rotated_point[1])


def translate(pose: StateSE2, translation: npt.NDArray[np.float64]) -> StateSE2:
    """ "
    Applies a 2D translation
    :param pose: The pose to be transformed
    :param translation: The translation to be applied
    :return: The translated pose
    """
    assert translation.shape == (2,) or translation.shape == (2, 1)
    return StateSE2(pose.x + translation[0], pose.y + translation[1], pose.heading)


def rotate(pose: StateSE2, rotation_matrix: npt.NDArray[np.float64]) -> StateSE2:
    """
    Applies a 2D rotation to an SE2 Pose
    :param pose: The pose to be transformed
    :param rotation_matrix: The 2x2 rotation matrix representing the rotation
    :return: The rotated pose
    """
    assert rotation_matrix.shape == (2, 2)
    rotated_point = np.array([pose.x, pose.y]) @ rotation_matrix
    rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[1, 1])
    return StateSE2(rotated_point[0], rotated_point[1], pose.heading + rotation_angle)


def rotate_angle(pose: StateSE2, theta: float) -> StateSE2:
    """
    Rotates the scene object by the given angle.
    :param pose: The input pose
    :param theta: The rotation angle.
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix: npt.NDArray[np.float64] = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return rotate(pose, rotation_matrix)


def transform(pose: StateSE2, transform_matrix: npt.NDArray[np.float64]) -> StateSE2:
    """
    Applies an SE2 transform
    :param pose: The input pose
    :param transform_matrix: The transform matrix, can be 2D (3x3) or 3D (4x4)
    """
    rotated_pose = rotate(pose, transform_matrix[:2, :2])
    return translate(rotated_pose, transform_matrix[:2, 2])


def translate_longitudinally(pose: StateSE2, distance: float) -> StateSE2:
    """
    Translate an SE2 pose longitudinally (along heading direction)
    :param pose: SE2 pose to be translated
    :param distance: [m] distance by which point (x, y, heading) should be translated longitudinally
    :return translated se2
    """
    translation: npt.NDArray[np.float64] = np.array([distance * np.cos(pose.heading), distance * np.sin(pose.heading)])
    return translate(pose, translation)


def translate_laterally(pose: StateSE2, distance: float) -> StateSE2:
    """
    Translate an SE2 pose laterally
    :param pose: SE2 pose to be translated
    :param distance: [m] distance by which point (x, y, heading) should be translated longitudinally
    :return translated se2
    """
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.array(
        [distance * np.cos(pose.heading + half_pi), distance * np.sin(pose.heading + half_pi)]
    )
    return translate(pose, translation)


def translate_longitudinally_and_laterally(pose: StateSE2, lon: float, lat: float) -> StateSE2:
    """
    Translate the position component of an SE2 pose longitudinally and laterally
    :param pose: SE2 pose to be translated
    :param lon: [m] distance by which a point should be translated in longitudinal direction
    :param lat: [m] distance by which a point should be translated in lateral direction
    :return Point2D translated position
    """
    half_pi = np.pi / 2.0
    translation: npt.NDArray[np.float64] = np.array(
        [
            (lat * np.cos(pose.heading + half_pi)) + (lon * np.cos(pose.heading)),
            (lat * np.sin(pose.heading + half_pi)) + (lon * np.sin(pose.heading)),
        ]
    )
    return translate(pose, translation)
