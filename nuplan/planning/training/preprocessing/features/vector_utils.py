from copy import deepcopy
from typing import Optional

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from nuplan.planning.script.builders.utils.utils_type import are_the_same_type, validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType


def _validate_coords_shape(coords: FeatureDataType) -> None:
    """
    Validate coordinates have proper shape: <num_map_elements, num_points_per_element, 2>.
    :param coords: Coordinates to validate.
    :raise ValueError: If coordinates dimensions are not valid.
    """
    if len(coords.shape) != 3 or coords.shape[2] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}. Expected shape: (*, *, 2)")


def rotate_coords(coords: npt.NDArray[np.float32], quaternion: Quaternion) -> npt.NDArray[np.float32]:
    """
    Rotate all vector coordinates within input tensor using input quaternion.
    :param coords: coordinates to translate: <num_map_elements, num_points_per_element, 2>.
    :param quaternion: Rotation to apply.
    :return rotated coords.
    """
    _validate_coords_shape(coords)
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


def translate_coords(
    coords: FeatureDataType, translation_value: FeatureDataType, avails: Optional[FeatureDataType] = None
) -> FeatureDataType:
    """
    Translate all vector coordinates within input tensor along x, y dimensions of input translation tensor.
        Note: Z-dimension ignored.
    :param coords: coordinates to translate: <num_map_elements, num_points_per_element, 2>.
    :param translation_value: <np.float: 3,>. Translation in x, y, z.
    :param avails: Optional mask to specify real vs zero-padded data to ignore in coords:
        <num_map_elements, num_points_per_element>.
    :return translated coords.
    :raise ValueError: If translation_value dimensions are not valid or coords and avails have inconsistent shape.
    """
    if translation_value.shape[0] != 3:
        raise ValueError(
            f"Translation value has incorrect dimensions: {translation_value.shape[0]}! Expected: 3 (x, y, z)"
        )
    _validate_coords_shape(coords)
    are_the_same_type(coords, translation_value)

    if avails is not None and coords.shape[:2] != avails.shape:
        raise ValueError(f"Mismatching shape between coords and availabilities: {coords.shape[:2]}, {avails.shape}")

    # apply translation
    coords = coords + translation_value[:2]

    # ignore zero-padded data
    if avails is not None:
        coords[~avails] = 0.0

    return coords


def scale_coords(coords: FeatureDataType, scale_value: FeatureDataType) -> FeatureDataType:
    """
    Scale all vector coordinates within input tensor along x, y dimensions of input scaling tensor.
        Note: Z-dimension ignored.
    :param coords: coordinates to scale: <num_map_elements, num_points_per_element, 2>.
    :param scale_value: <np.float: 3,>. Scale in x, y, z.
    :return scaled coords.
    :raise ValueError: If scale_value dimensions are not valid.
    """
    if scale_value.shape[0] != 3:
        raise ValueError(f"Scale value has incorrect dimensions: {scale_value.shape[0]}! Expected: 3 (x, y, z)")
    _validate_coords_shape(coords)
    are_the_same_type(coords, scale_value)

    return coords * scale_value[:2]


def xflip_coords(coords: FeatureDataType) -> FeatureDataType:
    """
    Flip all vector coordinates within input tensor along X-axis.
    :param coords: coordinates to flip: <num_map_elements, num_points_per_element, 2>.
    :return flipped coords.
    """
    _validate_coords_shape(coords)
    coords = deepcopy(coords)
    coords[:, :, 0] *= -1

    return coords


def yflip_coords(coords: FeatureDataType) -> FeatureDataType:
    """
    Flip all vector coordinates within input tensor along Y-axis.
    :param coords: coordinates to flip: <num_map_elements, num_points_per_element, 2>.
    :return flipped coords.
    """
    _validate_coords_shape(coords)
    coords = deepcopy(coords)
    coords[:, :, 1] *= -1

    return coords
