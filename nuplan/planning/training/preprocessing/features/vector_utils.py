import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from nuplan.planning.script.builders.utils.utils_type import are_the_same_type, validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType


def rotate_coords(coords: npt.NDArray[np.float64], quaternion: Quaternion) -> npt.NDArray[np.float64]:
    """
    Rotate all vector coordinates within input tensor using input quaternion.
    :param coords: coordinates to translate: <num_map_elements, num_points_per_element, 2>.
    :param quaternion: Rotation to apply.
    :return rotated coords.
    """
    validate_type(coords, np.ndarray)

    # Flatten the first two dimensions to make the shape (num_map_elements * num_points_per_element, 2).
    num_map_elements, num_points_per_element, _ = coords.shape
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    # Add zeros to the z dimension to make them 3D points.
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)

    # Rotate.
    coords = np.dot(quaternion.rotation_matrix.astype(coords.dtype), coords.T)

    # Remove the z dimension and reshape it back to (num_map_elements, num_points_per_element, 2).
    return coords.T[:, :2].reshape(num_map_elements, num_points_per_element, 2)  # type: ignore


def translate_coords(coords: FeatureDataType, translation_value: FeatureDataType) -> FeatureDataType:
    """
    Translate all vector coordinates within input tensor along x, y dimensions of input translation tensor.
        Note: Z-dimension ignored.
    :param coords: coordinates to translate: <num_map_elements, num_points_per_element, 2>.
    :param translation_value: <np.float: 3,>. Translation in x, y, z.
    :return translated coords.
    """
    assert translation_value.size == 3, "Translation value must have dimension of 3 (x, y, z)"
    are_the_same_type(coords, translation_value)

    return coords + translation_value[:2]


def scale_coords(coords: FeatureDataType, scale_value: FeatureDataType) -> FeatureDataType:
    """
    Scale all vector coordinates within input tensor along x, y dimensions of input scaling tensor.
        Note: Z-dimension ignored.
    :param coords: coordinates to scale: <num_map_elements, num_points_per_element, 2>.
    :param scale_value: <np.float: 3,>. Scale in x, y, z.
    :return scaled coords.
    """
    assert scale_value.size == 3, f"Scale value has incorrect dimension: {scale_value.size}!"
    are_the_same_type(coords, scale_value)

    return coords * scale_value[:2]


def xflip_coords(coords: FeatureDataType) -> FeatureDataType:
    """
    Flip all vector coordinates within input tensor along X-axis.
    :param coords: coordinates to flip: <num_map_elements, num_points_per_element, 2>.
    :return flipped coords.
    """
    coords[:, :, 0] *= -1
    return coords


def yflip_coords(coords: FeatureDataType) -> FeatureDataType:
    """
    Flip all vector coordinates within input tensor along Y-axis.
    :param coords: coordinates to flip: <num_map_elements, num_points_per_element, 2>.
    :return flipped coords.
    """
    coords[:, :, 1] *= -1
    return coords
