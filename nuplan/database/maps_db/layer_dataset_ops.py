from typing import Tuple

import numpy as np
import numpy.typing as npt


def get_transform_matrix(layer_dataset) -> npt.NDArray[np.float64]:  # type: ignore
    """
    Converts 2D affine.Affine objects to 3D numpy arrays.
    :param layer_dataset: A *context manager* for the layer dataset.
    :return: The transform matrix.
    """
    # Rasterio returns transforms as 2D affine.Affine objects (from https://pypi.org/project/affine/).
    # This function converts them to 3D numpy arrays, since that's what the MapLayer class expects.
    pixel_to_spatial = layer_dataset.transform
    # Example result from rasterio_dataset.transform:
    # | 0.10,  0.00, 363121.00 |
    # | 0.00, -0.10, 144938.00 |
    # | 0.00,  0.00, 1.00 |
    # Values are indexed indexed 0 to 8, left-to-right top-to-bottom.
    if pixel_to_spatial[1] != 0 or pixel_to_spatial[3] != 0:
        # See https://lists.osgeo.org/pipermail/gdal-dev/2016-August/045104.html
        raise ValueError(
            f"Rasterio dataset {layer_dataset.name} uses shear or rotation transform. "
            f"This is supposed to be impossible as GPKG standard only supports north-up. "
            f"Pixel to spatial transform was {pixel_to_spatial}"
        )
    # Invert the Affine object, since MapLayer needs a spatial to pixel transformation matrix.
    spatial_to_pixel = ~pixel_to_spatial

    return np.array(
        [
            [spatial_to_pixel[0], 0, 0, spatial_to_pixel[2]],
            [0, spatial_to_pixel[4], 0, spatial_to_pixel[5]],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def load_layer_as_numpy(layer_dataset, is_binary: bool) -> npt.NDArray[np.uint8]:  # type: ignore
    """
    Loads map layer data as a numpy array.
    :param layer_dataset: A *context manager* for the layer dataset.
    :param is_binary: Whether the layer is binary or not.
    :return: The layer data as numpy array.
    """
    if is_binary:
        raw_layer = layer_dataset.read(out_dtype=np.uint8)
        # Assume that the layer is multichannel, and the first channel is sufficient for binary use.
        layer_data = raw_layer[0, :, :]
        # layer_data often has all values set to 0 or 255. Make them set to 0 and 1 instead.
        layer_data[layer_data > 0] = 1
    else:
        raw_layer = layer_dataset.read()
        # Assume that the layer is multichannel, and that we can just take the first channel.
        layer_data = raw_layer[0, :, :]

    return np.array(layer_data)


def get_shape(layer_dataset) -> Tuple[float, float]:  # type: ignore
    """
    Gets the shape of the map layer.
    :param layer_dataset: A *context manager* for the layer dataset.
    :return: The height and width of the map layer.
    """
    return layer_dataset.height, layer_dataset.width
