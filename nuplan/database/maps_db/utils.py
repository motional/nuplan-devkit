import logging
import os
from mmap import PROT_READ, mmap
from typing import Dict, List, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import numpy.typing as npt

from nuplan.database.maps_db.metadata import MapLayerMeta

logger = logging.getLogger(__name__)


def load_mmap(path: str, size: Tuple[int, int], dtype: str) -> npt.NDArray[Union[np.uint8, np.float32]]:
    """
    Loads a binary file at path to a memory map and coverts to a numpy array.
    :param path: The path to load the binary file.
    :param size: The size of the numpy array.
    :param dtype: A string either 'int' or 'float'.
    :return: A mmap object.
    """
    assert dtype in {'int', 'float'}, f"Param dtype must be either int or float. Received {dtype}."
    if dtype == 'int':
        dtype = np.uint8  # type: ignore
    elif dtype == 'float':
        dtype = np.float32  # type: ignore

    with open(path, "rb") as fp:
        memory_map = mmap(fp.fileno(), 0, prot=PROT_READ)

    return np.ndarray(shape=size, dtype=dtype, buffer=memory_map)


def has_binary_masks(map_layer: MapLayerMeta, cache_dir: str) -> bool:
    """
    Checks if all binary masks are created.
    :param map_layer: A MapLayerMeta object.
    :param cache_dir: The directory to cache the binary mask.
    :return: True if binary masks are created, otherwise False.
    """
    binary_paths = [os.path.join(cache_dir, map_layer.binary_mask_name)]
    if map_layer.can_dilate:
        binary_paths.append(os.path.join(cache_dir, map_layer.binary_joint_dist_name))

    for binary_path in binary_paths:
        if not os.path.exists(binary_path) or os.path.getsize(binary_path) == 0:
            return False
    return True


def compute_joint_distance_matrix(array: npt.NDArray[np.uint8], precision: float) -> npt.NDArray[np.float64]:
    """
    For each pixel in `array`, computes the physical distance to the nearest
    mask boundary. Distances from a 0 to the boundary are returned as positive
    values, and distances from a 1 to the boundary are returned as negative
    values.
    :param array: Binary array of pixel values.
    :param precision: Meters per pixel.
    :return: The physical distance to the nearest mask boundary.
    """
    # Consider a location on a mask where there is a 0 next to a 1. The 0 is one
    # pixel away from the nearest 1, and the 1 is one pixel away from the
    # nearest 0. However, reporting that they are both distance 1 from the
    # *boundary* doesn't make sense spatially: by moving only *1* pixel, we
    # could go from "1px away from boundary" to "1px away from the boundary on
    # the other side.
    #
    # To resolve this, we decided that the boundary is halfway between the 0 and
    # 1, which we implement by subtracting 0.5px from the distance to the
    # nearest pixel of the opposite value. This doesn't work as intuitively for
    # diagonal distances, but eliminating the 'spatial discontinuity' is more
    # important than mostly arbitrary sub-pixel distance definitions.
    #
    # pixel values:                  0               1
    # distance from 0 to nearest 1:  |-------------->|
    # distance from 0 to boundary:   |------>|
    distances_0_to_boundary = cv2.distanceTransform((1.0 - array).astype(np.uint8), cv2.DIST_L2, 5)
    distances_0_to_boundary[distances_0_to_boundary > 0] -= 0.5
    distances_0_to_boundary = (distances_0_to_boundary * precision).astype(np.float32)

    distances_1_to_boundary = cv2.distanceTransform(array.astype(np.uint8), cv2.DIST_L2, 5)
    distances_1_to_boundary[distances_1_to_boundary > 0] -= 0.5
    distances_1_to_boundary = (distances_1_to_boundary * precision).astype(np.float32)

    return distances_0_to_boundary - distances_1_to_boundary  # type: ignore


def create_binary_masks(array: npt.NDArray[np.uint8], map_layer: MapLayerMeta, layer_dir: str) -> None:
    """
    Creates the binary mask for a given map layer in a given map version and
    stores it in the cache.
    :param array: Map array to write to binary.
    :param map_layer: Map layer to create the masks for.
    :param layer_dir: Directory where binary masks will be stored.
    """
    # Keep only one channel as they are all the same for all layers.
    if len(array.shape) == 3:
        array = array[:, :, 0]

    # The lidar intensity map uses shades of gray to capture differences
    # in the road surface (such as lane markings, crossings, and the edge of the road).
    # For this reason, we cannot convert the lidar intensity map into a binary 0,1 matrix.
    if map_layer.is_binary:
        array[array < 255] = 0
        array[array == 255] = 1

    destination = os.path.join(layer_dir, '{}')

    logger.debug('Writing binary mask to {}...'.format(destination.format(map_layer.binary_mask_name)))
    with open(destination.format(map_layer.binary_mask_name), "wb") as f:
        f.write(array.tobytes())
    logger.debug('Writing binary mask to {} done.'.format(destination.format(map_layer.binary_mask_name)))

    if map_layer.can_dilate:
        # Computing the distance from all points to the boundary
        logger.debug(
            'Writing joint distance mask to {}...'.format(destination.format(map_layer.binary_joint_dist_name))
        )
        joint_distances = compute_joint_distance_matrix(array, map_layer.precision)

        with open(destination.format(map_layer.binary_joint_dist_name), "wb") as f:
            f.write(joint_distances.tobytes())
        del joint_distances
        del array
        logger.debug(
            'Writing joint distance mask to {} done.'.format(destination.format(map_layer.binary_joint_dist_name))
        )


def connect_blp_predecessor(
    blp_id: str, lane_conn_info: gpd.geodataframe, cross_blp_conns: Dict[str, List[int]], ls_conns: List[List[int]]
) -> None:
    """
    Given a specific baseline path id, find its predecessor and update info in ls_connections information.
    :param blp_id: a specific baseline path id to query
    :param lane_conn_info: baseline paths information in intersections contains the from_blp/to_blp info
    :param cross_blp_conns: Dict to record the baseline path id as key(str) and [blp_start_ls_idx, blp_end_ls_idx] pair
        as value (List[int])
    :param ls_conns: lane_segment_connection to record the [from_ls_idx, to_ls_idx] connection info, updated with
        predecessors found.
    """
    blp_start, blp_end = cross_blp_conns[blp_id]
    predecessor_blp = lane_conn_info[lane_conn_info['to_blp'] == blp_id]
    predecessor_list = predecessor_blp['fid'].to_list()

    for predecessor_id in predecessor_list:
        predecessor_start, predecessor_end = cross_blp_conns[predecessor_id]
        ls_conns.append([predecessor_end, blp_start])


def connect_blp_successor(
    blp_id: str, lane_conn_info: gpd.geodataframe, cross_blp_conns: Dict[str, List[int]], ls_conns: List[List[int]]
) -> None:
    """
    Given a specific baseline path id, find its successor and update info in ls_connections information.
    :param blp_id: a specific baseline path id to query
    :param lane_conn_info: baseline paths information in intersections contains the from_blp/to_blp info
    :param cross_blp_conns: Dict to record the baseline path id as key(str) and [blp_start_ls_idx, blp_end_ls_idx] pair
        as value (List[int])
    :param ls_conns: lane_segment_connnection to record the [from_ls_idx, to_ls_idx] connection info, updated with
        predecessors found.
    """
    blp_start, blp_end = cross_blp_conns[blp_id]
    successor_blp = lane_conn_info[lane_conn_info['from_blp'] == blp_id]
    successor_list = successor_blp['fid'].to_list()

    for successor_id in successor_list:
        successor_start, successor_end = cross_blp_conns[successor_id]
        ls_conns.append([blp_end, successor_start])


def build_lane_segments_from_blps(
    candidate_blps: gpd.geodataframe,
    ls_coords: List[List[List[float]]],
    ls_conns: List[List[int]],
    ls_groupings: List[List[int]],
    cross_blp_conns: Dict[str, List[int]],
) -> None:
    """
    Process candidate baseline paths to small portions of lane-segments with connection info recorded.
    :param candidate_blps: Candidate baseline paths to be cut to lane_segments
    :param ls_coords: Output data recording lane-segment coordinates in format of [N, 2, 2]
    :param ls_conns: Output data recording lane-segment connection relations in format of [M, 2]
    :param ls_groupings: Output data recording lane-segment indices associated with each lane in format
        [num_lanes, num_segments_in_lane]
    :param: cross_blp_conns: Output data recording start_idx/end_idx for each baseline path with id as key.
    """
    for _, blp in candidate_blps.iterrows():
        blp_id = blp['fid']
        px, py = blp.geometry.coords.xy
        ls_num = len(px) - 1
        blp_start_ls = len(ls_coords)
        blp_end_ls = blp_start_ls + ls_num - 1
        ls_grouping = []
        for idx in range(ls_num):
            curr_pt, next_pt = [px[idx], py[idx]], [px[idx + 1], py[idx + 1]]
            ls_idx = len(ls_coords)
            if idx > 0:
                ls_conns.append([ls_idx - 1, ls_idx])
            ls_coords.append([curr_pt, next_pt])
            ls_grouping.append(ls_idx)
        ls_groupings.append(ls_grouping)
        cross_blp_conns[blp_id] = [blp_start_ls, blp_end_ls]
