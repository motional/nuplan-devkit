from __future__ import annotations

import logging
import math
from bisect import bisect_right
from functools import reduce
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from shapely import geometry

from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.utils.boxes.box3d import Box3D, box_in_image
from nuplan.database.utils.geometry import view_points
from nuplan.database.utils.label.label import Label
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

# Importing directly would lead to circular imports.
NuPlanDB = "NuPlanDB"
Image = "Image"
LidarPc = "LidarPc"
EgoPose = "EgoPose"
LidarBox = "LidarBox"

logger = logging.getLogger(__name__)


def generate_multi_scale_connections(
    connections: npt.NDArray[np.float64], scales: List[int]
) -> Dict[int, npt.NDArray[np.float64]]:
    """
    Generate multi-scale connections by finding the neighors up to max(scales) hops away for each node.

    :param connections: <np.float: num_connections, 2>. 1-hop connections.
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
    """
    # This dict will have format {node_idx: neighbor_dict},
    # where each neighbor_dict will have format {'i_hop_neighbors': set_of_i_hop_neighbors}.
    node_idx_to_neighbor_dict: Dict[int, Dict[str, Set[int]]] = {}

    # Initialize the data structure for each node with its 1-hop neighbors.
    for connection in connections:
        start_idx, end_idx = list(connection)
        if start_idx not in node_idx_to_neighbor_dict:
            node_idx_to_neighbor_dict[start_idx] = {"1_hop_neighbors": set()}
        if end_idx not in node_idx_to_neighbor_dict:
            node_idx_to_neighbor_dict[end_idx] = {"1_hop_neighbors": set()}
        node_idx_to_neighbor_dict[start_idx]["1_hop_neighbors"].add(end_idx)

    # Find the neighors up to max(scales) hops away for each node.
    for scale in range(2, max(scales) + 1):
        for neighbor_dict in node_idx_to_neighbor_dict.values():
            neighbor_dict[f"{scale}_hop_neighbors"] = set()
            for n_hop_neighbor in neighbor_dict[f"{scale - 1}_hop_neighbors"]:
                for n_plus_1_hop_neighbor in node_idx_to_neighbor_dict[n_hop_neighbor]["1_hop_neighbors"]:
                    neighbor_dict[f"{scale}_hop_neighbors"].add(n_plus_1_hop_neighbor)

    # Get the connections of each scale.
    multi_scale_connections: Dict[int, npt.NDArray[np.float64]] = {}
    for scale in scales:
        scale_connections = []
        for node_idx, neighbor_dict in node_idx_to_neighbor_dict.items():
            for n_hop_neighbor in neighbor_dict[f"{scale}_hop_neighbors"]:
                scale_connections.append([node_idx, n_hop_neighbor])
        multi_scale_connections[scale] = np.array(scale_connections)

    return multi_scale_connections


def get_boxes(
    sample_data: Union[LidarPc, Image],  # type: ignore
    frame: Frame = Frame.GLOBAL,
    trans_matrix_ego: Optional[npt.NDArray[np.float64]] = None,
    trans_matrix_sensor: Optional[npt.NDArray[np.float64]] = None,
) -> List[Box3D]:
    """
    Given a LidarPc/Image record, this function returns a list of boxes (in the global coordinate frame by
    default) associated with that record. It simply converts the annotations to boxes.
    :param sample_data: Either a Lidar pointcloud or an Image.
        Note: Having the type Union[LidarPc, Image] for this throws error for TRT with Python 3.8.
    :param frame: An enumeration of Frame (global/vehicle/sensor).
    :param trans_matrix_ego:
        Transformation matrix to transform the boxes from the global frame to the ego-vehicle frame.
    :param trans_matrix_sensor:
        Transformation matrix to transform the boxes from the ego-vehicle frame to the sensor frame.
    :return: List of boxes in the global coordinate frame.
    """
    if frame == Frame.VEHICLE:
        assert trans_matrix_ego is not None

    if frame == Frame.SENSOR:
        assert trans_matrix_ego is not None
        assert trans_matrix_sensor is not None

    boxes = [sa.box() for sa in sample_data.lidar_boxes]  # type: ignore

    if frame in [Frame.VEHICLE, Frame.SENSOR]:
        for box in boxes:
            box.transform(trans_matrix_ego)

    if frame == Frame.SENSOR:
        for box in boxes:
            box.transform(trans_matrix_sensor)

    return boxes


def add_future_boxes(
    tracks_dict: Dict[str, List[Box3D]],
    lidar_pcs: List[LidarPc],  # type: ignore
) -> Dict[str, List[Box3D]]:
    """
    Iterate over future samples, adding boxes to the box sequence associated with each track token
    :param tracks_dict: Dictionary of boxes associated with track tokens.
    :param lidar_pcs: List of LidarPc.
    :return: Updated Dictionary that includes future boxes.
    """
    for idx, next_lidar_pc in enumerate(lidar_pcs[1:]):
        next_lidar_boxes = next_lidar_pc.lidar_boxes  # type: ignore
        for lidar_box in next_lidar_boxes:
            track_token = lidar_box.track_token
            # Add tracks for each box that appears in the next sample
            if track_token not in tracks_dict:
                # Skip boxes that were not in the original t0 sample
                continue
            tracks_dict[track_token].append(lidar_box.box())

        # Add "None" boxes to box sequences that do not appear at this timestamp
        for track_token, track in tracks_dict.items():
            if len(track) < idx + 1:
                tracks_dict[track_token].append(None)

    return tracks_dict


def get_future_box_sequence(
    lidar_pcs: List[LidarPc],  # type: ignore
    frame: Frame,
    future_horizon_len_s: float,
    future_interval_s: float,
    extrapolation_threshold_ms: float = 1e5,
    trans_matrix_ego: Optional[npt.NDArray[np.float64]] = None,
    trans_matrix_sensor: Optional[npt.NDArray[np.float64]] = None,
) -> Dict[str, List[Box3D]]:
    """
    Get a mapping from track token to box sequence over time for each box in the input data. Box
    annotations are sampled at a frequency of 20Hz.
    :param lidar_pcs: List of LidarPc.
    :param frame: An enumeration of Frame (global/vehicle/sensor).
    :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
    :param future_interval_s: Timestamp interval of the future waypoints in seconds.
    :param extrapolation_threshold_ms: If a target interpolation timestamp extends beyond the timestamp of the
        last recorded bounding box for an actor, then the values for the box position at the target timestamp will only
        be extrapolated if the target timestamp is within the specified number of microseconds of the last recorded
        bounding box. Otherwise the box at the target timestamp will be set to None.
    :param trans_matrix_ego:
        Transformation matrix to transform the boxes from the global frame to the ego-vehicle frame.
    :param trans_matrix_sensor:
        Transformation matrix to transform the boxes from the ego-vehicle frame to the sensor frame.
    :return: Mapping from track token to list of corresponding boxes at each timestamp in the global coordinate
        frame, where the first box corresponds to the current timestamp t0.
    """
    if frame == Frame.VEHICLE:
        assert trans_matrix_ego is not None

    if frame == Frame.SENSOR:
        assert trans_matrix_ego is not None
        assert trans_matrix_sensor is not None

    num_future_boxes = int(future_horizon_len_s / future_interval_s)
    num_target_timestamps = num_future_boxes + 1  # include box at current frame t0
    future_horizon_len_ms = future_horizon_len_s * 1e6
    start_timestamp = lidar_pcs[0].timestamp  # type: ignore

    tracks_dict = {lidar_box.track_token: [lidar_box.box()] for lidar_box in lidar_pcs[0].lidar_boxes}  # type: ignore

    timestamps = [lidar_pc.timestamp for lidar_pc in lidar_pcs]  # type: ignore

    tracks_dict = add_future_boxes(tracks_dict, lidar_pcs)

    # Interpolate box positions at specified time intervals
    target_timestamps: Union[npt.NDArray[np.float64], List[float]] = np.linspace(
        start=start_timestamp, stop=start_timestamp + future_horizon_len_ms, num=num_target_timestamps
    )

    for track_token, track in tracks_dict.items():
        last_box_index = get_last_box_index(box_sequence=track)
        if last_box_index == 0:
            # All future boxes are None and we do not want to extrapolate, so we can exit this iteration early
            tracks_dict[track_token] = [track[0]] + [None] * num_future_boxes
            continue
        last_box_timestamp = timestamps[last_box_index]
        target_timestamps = [t for t in target_timestamps if t <= last_box_timestamp + extrapolation_threshold_ms]

        box_indices = [i for i, box in enumerate(track) if box is not None]
        interpolated_boxes: List[Optional[Box3D]] = []
        interpolated_boxes.extend(
            interpolate_boxes(
                target_timestamps=target_timestamps,
                timestamps=np.array([float(timestamps[i]) for i in box_indices]),
                box_sequence=[track[i] for i in box_indices],
            )
        )

        # Transform box to ego coordinates
        if frame in [Frame.VEHICLE, Frame.SENSOR]:
            for box in interpolated_boxes:
                if box is not None:
                    box.transform(trans_matrix_ego)

        if frame == Frame.SENSOR:
            for box in interpolated_boxes:
                if box is not None:
                    box.transform(trans_matrix_sensor)

        # Pad missing boxes at the end of the sequence with None values
        num_missing_final_boxes = num_target_timestamps - len(interpolated_boxes)
        if num_missing_final_boxes:
            interpolated_boxes.extend([None] * num_missing_final_boxes)
        tracks_dict[track_token] = interpolated_boxes

    return tracks_dict


def get_last_box_index(box_sequence: List[Box3D]) -> int:
    """
    Given a list of boxes, find the highest index such that the value of the box at that index is not None.
    :param box_sequence: Sequence of boxes, for example, representing box positions over time.
    :return: List index representing the highest index such that the value of the box at that index is not None.
    """
    for i in reversed(range(len(box_sequence))):
        if box_sequence[i] is not None:
            return i
    raise ValueError(f"All boxes in sequence are None: {box_sequence}")


def interpolate_boxes(
    target_timestamps: Union[npt.NDArray[np.float64], List[float]],
    timestamps: Union[npt.NDArray[np.float64], List[float]],
    box_sequence: List[Box3D],
) -> List[Box3D]:
    """
    Given a sequence of boxes representing box positions over time along with their corresponding timestamps,
    interpolate the box center, rotation, velocity and angular velocity at the target timestamps. Target timestamps
    should lie within the range of the raw timestamps corresponding to recorded data.
    :param target_timestamps: Times at which box values will be interpolated, sorted in increasing order.
    :param timestamps: Times corresponding to each box in the box sequence, sorted in increasing order.
    :param box_sequence: Sequence of boxes at each timestamp corresponding to an actor's position over time, from the
        raw data. The first box corresponds to the current frame.
    :return: Sequence of boxes at each timestamp corresponding to an actor's position over time, where box center,
        rotation, velocity and angular velocity are interpolated at the target timestamps.
    """
    assert len(timestamps) == len(box_sequence)
    if sorted(list(timestamps)) != list(timestamps) or sorted(list(target_timestamps)) != list(target_timestamps):
        raise ValueError(
            f"Check input. Timestamps should be sorted, but received the following inputs:\n"
            f"timestamps: {list(timestamps)}\ntarget_timestamps: {target_timestamps}"
        )
    centers = interpolate_coordinates(
        box_timestamps=timestamps,
        box_coordinates=[box.center for box in box_sequence],
        target_timestamps=target_timestamps,
    )
    rotations = interpolate_rotations(
        boxes=box_sequence, box_timestamps=list(timestamps), target_timestamps=list(target_timestamps)
    )
    velocities = interpolate_coordinates(
        box_timestamps=timestamps,
        box_coordinates=[box.velocity for box in box_sequence],
        target_timestamps=target_timestamps,
    )

    angular_velocities = np.interp(
        x=target_timestamps, xp=timestamps, fp=np.array([box.angular_velocity for box in box_sequence])
    )

    first_box = box_sequence[0]
    interpolated_boxes = []
    for c, r, v, w in zip(centers, rotations, velocities, angular_velocities):
        next_box = first_box.copy()
        next_box.center, next_box.orientation, next_box.velocity, next_box.angular_velocity = c, r, v, w
        interpolated_boxes.append(next_box)

    return interpolated_boxes


def interpolate_rotations(
    boxes: List[Box3D], box_timestamps: List[float], target_timestamps: List[float]
) -> List[Quaternion]:
    """
    Given a sequence of boxes representing box positions over time along with their corresponding timestamps,
    interpolate the box rotation at the target timestamps. Target timestamps should lie within
    the range of the raw timestamps corresponding to recorded data.
    :param target_timestamps: Times at which box values will be interpolated, sorted in increasing order.
    :param box_timestamps: Times corresponding to each box in the box sequence, sorted in increasing order.
    :param boxes: Sequence of boxes at each timestamp corresponding to an actor's position over time, from the
        raw data. The first box corresponds to the current frame.
    :return: Sequence of Quaternion rotations at each timestamp corresponding to an actor's rotation over time, where
        rotation is interpolated at the target timestamps.
    """
    rotations = []
    for current_target_timestamp in target_timestamps:
        next_box_index = bisect_right(box_timestamps[:-1], current_target_timestamp)
        prev_box_index = next_box_index - 1
        assert prev_box_index >= 0
        tdiff_between_boxes = box_timestamps[next_box_index] - box_timestamps[prev_box_index]
        assert tdiff_between_boxes > 0.0
        target_tdiff_from_prev = current_target_timestamp - box_timestamps[prev_box_index]
        target_timestamp_relative_position = target_tdiff_from_prev / tdiff_between_boxes
        interpolated_box_rotation = Quaternion.slerp(
            q0=boxes[prev_box_index].orientation,
            q1=boxes[next_box_index].orientation,
            amount=target_timestamp_relative_position,
        )
        rotations.append(interpolated_box_rotation)
    return rotations


def interpolate_coordinates(
    target_timestamps: Union[npt.NDArray[np.float64], List[float]],
    box_timestamps: Union[npt.NDArray[np.float64], List[float]],
    box_coordinates: List[Union[npt.NDArray[np.float64], Tuple[float, ...]]],
) -> List[npt.NDArray[np.float64]]:
    """
    Given a sequence of boxes representing box positions over time along with their corresponding timestamps,
    interpolate the box coordinates at the target timestamps. Target timestamps should lie within
    the range of the raw timestamps corresponding to recorded data.
    :param target_timestamps: Times at which box coordinates will be interpolated, sorted in increasing order.
    :param box_timestamps: Times corresponding to each box coordinate in the box sequence, sorted in increasing order.
    :param box_coordinates: Sequence of box coordinates at each timestamp corresponding to an actor's position over
        time, from the raw data. The first box corresponds to the current frame.
    :return: Sequence of array coordinate positions in np.array(x, y, z) format at each timestamp corresponding to an
        actor's position over time, where position is interpolated at the target timestamps.
    """
    xs = list(
        np.interp(
            x=target_timestamps, xp=box_timestamps, fp=np.array([coordinate[0] for coordinate in box_coordinates])
        )
    )
    ys = list(
        np.interp(
            x=target_timestamps, xp=box_timestamps, fp=np.array([coordinate[1] for coordinate in box_coordinates])
        )
    )
    zs = list(
        np.interp(
            x=target_timestamps, xp=box_timestamps, fp=np.array([coordinate[2] for coordinate in box_coordinates])
        )
    )
    centers = [np.array([x, y, z]) for x, y, z in zip(xs, ys, zs)]  # type: ignore
    return centers


def pack_future_boxes(
    track_token_2_box_sequence: Dict[str, List[Box3D]], future_horizon_len_s: float, future_interval_s: float
) -> List[Box3D]:
    """
    Given a mapping from all the track tokens to the list of corresponding boxes at each future
        timestamp, this function packs the "future" data into the individual Box3D boxes in the current Sample,
        such that each box contains its future center positions and future orientations in subsequent frames.
    :param track_token_2_box_sequence: Mapping from track token to list of corresponding boxes at each timestamp
        in the global coordinate frame, returned by function get_future_box_sequence()
    :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
    :param future_interval_s: Timestamp interval of the future waypoints in seconds.
    :return: List of boxes in a frame, where each box contains future center positions and future orientations in
        subsequent frames.
    """
    boxes_out: List[Box3D] = []
    for track_token, box_sequence in track_token_2_box_sequence.items():
        current_box = box_sequence[0]
        future_centers = [[box.center if box else (np.nan, np.nan, np.nan) for box in box_sequence[1:]]]
        future_orientations = [[box.orientation if box else None for box in box_sequence[1:]]]
        mode_probs = [1.0]
        box_with_future = Box3D(
            center=current_box.center,
            size=current_box.size,
            orientation=current_box.orientation,
            label=current_box.label,
            score=current_box.score,
            velocity=current_box.velocity,
            angular_velocity=current_box.angular_velocity,
            payload=current_box.payload,
            token=current_box.token,
            track_token=current_box.track_token,
            future_horizon_len_s=future_horizon_len_s,
            future_interval_s=future_interval_s,
            future_centers=future_centers,
            future_orientations=future_orientations,
            mode_probs=mode_probs,
        )
        boxes_out.append(box_with_future)
    return boxes_out


def transform_ego_traj(
    ego_poses: npt.NDArray[np.float64], transform_matrix: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Transform the ego trajectory to the first ego pose.
    :param ego_poses: Ego trajectory to transform.
    :param transform_matrix: Transformation to apply.
    :return The transformed ego poses.
    """
    ego_poses_new = transform_matrix[:3, :3] @ ego_poses[:, 0:3].T + transform_matrix[:3, 3].reshape((-1, 1))
    ego_poses[:, 0:3] = ego_poses_new.T

    return ego_poses


def get_future_ego_trajectory(
    lidarpc_rec: LidarPc,  # type: ignore
    future_ego_poses: List[EgoPose],  # type: ignore
    transformmatrix: npt.NDArray[np.float64],
    future_horizon_len_s: float,
    future_interval_s: float = 0.5,
    extrapolation_threshold_ms: float = 1e5,
) -> npt.NDArray[np.float64]:
    """
    Extract ego trajectory data starting from current sample timestamp for a duration of
        future horizon length in seconds.
    :param lidarpc_rec: Lidar point cloud record.
    :param future_ego_poses: future ego poses for a duration of horizon length.
    :param transformmatrix: Transformation matrix to transform the boxes from the global frame to the map_crop frame.
    :param future_horizon_len_s: Timestamp horizon of the future waypoints in seconds.
    :param future_interval_s: Timestamp interval of the future waypoints in seconds.
    :param extrapolation_threshold_ms: If the ego interpolation timestamp extends beyond the timestamp of the
        last recorded pose for the ego, then the values for the box position at the target timestamp will only
        be extrapolated if the target timestamp is within the specified number of microseconds of the last recorded
        pose. Otherwise the pose at the target timestamp will be set to None.
    :return: 2d numpy array of extracted trajectory data. Columns are
        (x_map, y_map, z_map, timestamp)
    """
    num_future_poses = int(future_horizon_len_s / future_interval_s)
    num_target_timestamps = num_future_poses + 1  # Include box at current frame t0

    start_timestamp = lidarpc_rec.ego_pose.timestamp  # type: ignore

    ego_traj: List[Tuple[float, ...]] = [(lidarpc_rec.ego_pose.x, lidarpc_rec.ego_pose.y, lidarpc_rec.ego_pose.z)]  # type: ignore
    timestamps = [start_timestamp]
    ego_traj.extend([(pose.x, pose.y, pose.z) for pose in future_ego_poses])  # type: ignore
    timestamps.extend([pose.timestamp for pose in future_ego_poses])  # type: ignore

    # We want to get ego pose at target timestamps, so we have to interpolate.
    target_timestamps: Union[npt.NDArray[np.float64], List[float]] = np.linspace(
        start=start_timestamp, stop=start_timestamp + future_horizon_len_s * 1e6, num=num_target_timestamps
    )
    # The ego trajectory may be smaller because of the end of extraction.
    last_ego_timestamp = timestamps[-1]
    target_timestamps = [t for t in target_timestamps if t <= last_ego_timestamp + extrapolation_threshold_ms]

    interpolated_ego_traj = interpolate_coordinates(
        target_timestamps=target_timestamps,
        box_timestamps=np.array([float(ts) for ts in timestamps]),
        box_coordinates=ego_traj,  # type: ignore
    )

    # (x, y, z, time) waypoint dims
    ego_traj_np = np.zeros((len(interpolated_ego_traj), 4))
    for i, wp in enumerate(interpolated_ego_traj):
        ego_traj_np[i, :] = [wp[0], wp[1], wp[2], target_timestamps[i]]

    num_waypoint = ego_traj_np.shape[0]
    if num_waypoint < num_target_timestamps:
        num_missing_rows = num_target_timestamps - num_waypoint
        padded_row = np.array([np.nan, np.nan, np.nan, np.nan])  # type: ignore
        padding = np.tile(padded_row, (num_missing_rows, 1))  # type: ignore
        ego_traj_np = np.concatenate((ego_traj_np, padding), axis=0)

    ego_poses = transform_ego_traj(ego_traj_np, lidarpc_rec.ego_pose.trans_matrix_inv)  # type: ignore

    transf_matrix = transformmatrix.astype(np.float32)  # type: ignore
    ego_poses = transformmatrix[:3, :3] @ ego_traj_np[:, 0:3].T + transf_matrix[:3, 3].reshape((-1, 1))
    ego_traj_np[:, 0:3] = ego_poses.T

    return ego_traj_np


def get_candidates(
    position: Union[Tuple[float, float], npt.NDArray[np.float64]],
    xrange: Union[Tuple[float, float], npt.NDArray[np.float64]],
    yrange: Union[Tuple[float, float], npt.NDArray[np.float64]],
    lane_groups_gdf: gpd.geodataframe,
    intersections_gdf: gpd.geodataframe,
) -> Tuple[gpd.geodataframe, gpd.geodataframe]:
    """
    Given a sample ego_pose position, find applicable lane_groups and intersections within its range.
    :param position: Ego pose position.
    :param xrange: only inside or intersects with xrange would lane_groups and intersections be considered.
    :param yrange: only inside or intersects with yrange would lane_groups and intersections be considered.
    :param lane_groups_gdf: dataframe of lane_groups data
    :param intersections_gdf: dataframe of intersections data
    :return: selected lane_groups dataframe and intersections dataframe within the range of sample ego-pose.
    """
    x_min, x_max = position[0] + xrange[0], position[0] + xrange[1]
    y_min, y_max = position[1] + yrange[0], position[1] + yrange[1]

    patch = geometry.box(x_min, y_min, x_max, y_max)
    candidate_lane_groups = lane_groups_gdf[lane_groups_gdf["geometry"].intersects(patch)]
    candidate_intersections = intersections_gdf[intersections_gdf["geometry"].intersects(patch)]

    return candidate_lane_groups, candidate_intersections


def render_pc(
    sample_data: LidarPc,  # type: ignore
    with_anns: bool = True,
    view_3d: npt.NDArray[np.float64] = np.eye(4),
    axes_limit: float = 40,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> None:
    """
    This is a naive rendering of the Lidar pointclouds with appropriate boxes. This is naive in the sense that it
    only renders the points but not the velocity associated with those points.
    :param sample_data: The Lidar pointcloud.
        Note: Having the type Union[LidarPc] for this throws error for TRT with Python 3.5.
    :param with_anns: Whether you want to render the annotations?
    :param view_3d: <np.float: 4, 4>. Define a projection needed (e.g. for drawing projection in an image).
    :param axes_limit: The range of that will be rendered will be between (-axes_limit, axes_limit).
    :param ax: Axes object or array of Axes objects.
    :param title: Title of the plot you want to render.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))
    points = view_points(sample_data.load().points[:3, :], view_3d, normalize=False)  # type: ignore
    ax.scatter(points[0, :], points[1, :], c=points[2, :], s=2)
    if with_anns:
        for box in sample_data.boxes(Frame.SENSOR):  # type: ignore
            # Get the LidarBox record with the same token as box.payload
            ann_record = sample_data.lidar_box[box.payload]  # type: ignore

            if not ann_record.track:
                logger.error("Wrong 3d instance mapping", ann_record)
                c: npt.NDArray[np.float64] = np.array([128, 0, 128]) / 255.0
            else:
                c = ann_record.track.category.color_np
            color = c, c, np.array([0, 0, 0])  # type: ignore
            box.render(ax, view=view_3d, colors=color)
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    ax.set_title("{}".format(title))


def translate(inp: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Translate a vector.
    :param inp: Vector to translate.
    :param x: Translation.
    :return: Translated vector.
    """
    return inp + x


def rotate(inp: npt.NDArray[np.float64], quaternion: Quaternion) -> npt.NDArray[np.float64]:
    """
    Rotate a vector.
    :param inp: Vector to rotate.
    :param quaternion: Rotation.
    :return: Rotated vector.
    """
    rotation_matrix: npt.NDArray[np.float64] = quaternion.rotation_matrix
    return np.dot(rotation_matrix, inp)  # type: ignore


def transform(inp: npt.NDArray[np.float64], trans_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Transform a vector.
    :param inp: Vector to transform.
    :param trans_matrix: Transformation matrix.
    :return: Transformed vector.
    """
    inp = rotate(inp, Quaternion(matrix=trans_matrix[:3, :3]))
    inp = translate(inp, trans_matrix[:3, 3])
    return inp


def scale(inp: npt.NDArray[np.float64], scale: Tuple[float, float, float]) -> npt.NDArray[np.float64]:
    """
    Scale a vector.
    :param inp: Vector to scale.
    :param scale: Scale factors.
    :return: Scaled vector.
    """
    scale_np = np.asarray(scale)  # type: ignore
    assert len(scale_np) == 3
    return inp * scale_np


def get_colors_marker(
    labelmap: Optional[Dict[int, Label]], box: Box3D
) -> Tuple[Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], str]], Optional[str]]:
    """
    Get the color and marker to use.
    :param labelmap: The labelmap is used to color the boxes. If not provided, default colors from box.render() will be
        used.
    :param box: The box for which color and marker are to be returned.
    :return: The color and marker to be used.
    """
    if labelmap is not None:
        c = np.array(labelmap[box.label].color)[:-1] / 255.0
        colors = (c, c, "k")
    else:
        colors = None

    if box.label == 2:  # ped
        marker = None
    else:
        marker = "o"

    return colors, marker


def draw_line(
    canvas: Union[npt.NDArray[np.float64], Axes],
    from_x: int,
    to_x: int,
    from_y: int,
    to_y: int,
    color: Tuple[Union[int, float], Union[int, float], Union[int, float]],
    linewidth: float,
    marker: Optional[str] = None,
    alpha: float = 1.0,
) -> None:
    """
    Draw a line on a matplotlib/cv2 canvas. Note that marker is not used in cv2.
    :param canvas: Canvas to draw.
    :param from_x: Start x position.
    :param to_x: End x position.
    :param from_y: Start y position.
    :param to_y: End y position.
    :param color: Color of the line.
    :param linewidth: Width of the line.
    :param marker: Marker to use, defaults to None
    :param alpha: Alpha channel of the color, defaults to 1.0
    """
    if isinstance(canvas, np.ndarray):
        color_int = tuple(int(c * 255) for c in color)
        cv2.line(canvas, (int(from_x), int(from_y)), (int(to_x), int(to_y)), color_int[::-1], linewidth)
    else:
        canvas.plot([from_x, to_x], [from_y, to_y], color=color, linewidth=linewidth, marker=marker, alpha=alpha)


def draw_future_ego_poses(
    ego_box: Box3D,
    ego_poses_np: npt.NDArray,  # type: ignore
    color: Tuple[float, float, float],
    ax: Union[npt.NDArray[np.float64], Axes],
) -> None:
    """
    Draw Future Ego Poses
    :param ego_box: Ego Vehicle Box.
    :param ego_poses_np: Numpy array containing future Ego Poses.
    :param color: Color to use.
    :param ax: Canvas to draw.
    """
    prev_x, prev_y = ego_box.center[0], ego_box.center[1]
    for idx in range(1, ego_poses_np.shape[0]):
        next_x, next_y = ego_poses_np[idx, 0], ego_poses_np[idx, 1]
        alpha = max(1.0 - idx * 0.1, 0.1)
        draw_line(
            from_x=prev_x,
            to_x=next_x,
            from_y=prev_y,
            to_y=next_y,
            color=color,
            marker="o",
            linewidth=1.0,
            canvas=ax,
            alpha=alpha,
        )
        prev_x, prev_y = next_x, next_y


def render_on_map(
    lidarpc_rec: LidarPc,  # type: ignore
    db: NuPlanDB,  # type: ignore
    boxes_lidar: List[Box3D],
    ego_poses: List[EgoPose],  # type: ignore
    points_to_render: Optional[npt.NDArray[np.float64]] = None,
    radius: float = 80.0,
    ax: Axes = None,
    labelmap: Optional[Dict[int, Label]] = None,
    render_boxes_with_velocity: bool = False,
    render_map_raster: bool = False,
    render_vector_map: bool = False,
    track_token: Optional[str] = None,
    with_random_color: bool = False,
    render_future_ego_poses: bool = False,
) -> plt.axes:
    """
    This function is used to render a LidarPC and boxes (in the lidar frame) on the map.
    :param lidarpc_rec: LidarPc record from NuPlanDB.
    :param db: Log database.
    :param boxes_lidar: List of boxes in the lidar frame.
    :param ego_poses: Ego poses to render.
    :param points_to_render: <np.float: nbr_indices, nbr_points>. If the user wants to visualize only a specific set
        of points (example points from selective rings/drivable area filtered/...) and not the entire pointcloud, they
        can pass those points along. Note that nbr_indices >=2 i.e. the user should at least pass (x, y).
    :param radius: The radius (centered on the Lidar) outside which we won't keep any points or boxes.
    :param ax: Axis on which to render.
    :param labelmap: The labelmap is used to color the boxes. If not provided, default colors from box.render() will be
        used.
    :param render_boxes_with_velocity: Whether you want to show the velocity arrow when you render the box?
    :param render_map_raster: Boolean indicating whether to include visualization of map layers from rasterized map.
    :param render_vector_map: Boolean indicating whether to include visualization of baseline paths from vector map.
    :param track_token: Which track to render, if it's None, render all the tracks.
    :param with_random_color: Whether to render the instances with different random color.
    :param render_future_ego_poses: Whether to render future EgoPoses.
    :return: plt.axes corresponding to BEV image with specified visualizations.
    """
    xrange = (-radius, radius)
    yrange = (-radius, radius)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

    (intensity_map_crop, intensity_map_translation, intensity_map_scale,) = lidarpc_rec.ego_pose.get_map_crop(  # type: ignore
        db.maps_db, xrange, yrange, "intensity", rotate_face_up=True  # type: ignore
    )

    map_translation = intensity_map_translation
    map_scale = intensity_map_scale

    lidar_to_ego = lidarpc_rec.lidar.trans_matrix  # type: ignore
    ego_to_global = lidarpc_rec.ego_pose.trans_matrix  # type: ignore

    # We need a transformation for points and boxes to compensate with the map_crop rotation in ego_pose.get_map_crop
    # To get the transformation, we need to isolate rotation around z-axis from the Quaternion.
    # Construct scipy rotation instance using rotation_matrix from quaternion.
    map_align_rot = R.from_matrix(lidarpc_rec.ego_pose.quaternion.rotation_matrix.T)  # type: ignore

    # Extract the rotation angle around 'z' axis. After this rotation, the points and boxes are oriented with
    # ego_vehicle facing right. To make it face up, we add extra pi/2 rotation.
    map_align_rot_angle = map_align_rot.as_euler("zxy")[0] + (math.pi / 2)
    map_align_transform = Quaternion(axis=[0, 0, 1], angle=map_align_rot_angle).transformation_matrix

    if render_map_raster:
        map_raster, map_translation, map_scale = lidarpc_rec.ego_pose.get_map_crop(  # type: ignore
            maps_db=db.maps_db,  # type: ignore
            xrange=xrange,
            yrange=yrange,
            map_layer_name="drivable_area",
            rotate_face_up=True,
        )
        ax.imshow(map_raster[::-1, :], cmap="gray")
    else:
        if intensity_map_crop is not None:
            ax.imshow(intensity_map_crop[::-1, :], cmap="gray")

    # Maps are mirrored compared to global coord-system, so we flip the maps.
    # We then flip the y-axis to return the maps to the right orientation.
    # This is only needed if we actually have found maps.
    if intensity_map_crop is not None:
        ax.set_ylim(ax.get_ylim()[::-1])

    pointcloud = lidarpc_rec.load(db)  # type: ignore

    if points_to_render is not None:
        pointcloud.points = points_to_render

    # Filter points that fall outside the radius.
    keep = np.sqrt(pointcloud.points[0] ** 2 + pointcloud.points[1] ** 2) < radius
    pointcloud.points = pointcloud.points[:, keep]

    global_to_crop = np.array(
        [
            [map_scale[0], 0, 0, map_translation[0]],
            [0, map_scale[1], 0, map_translation[1]],
            [0, 0, map_scale[2], 0],
            [0, 0, 0, 1],
        ]
    )  # type: ignore

    # Since point clouds are in float32 and global coordinates can have values
    # over 4 million, we avoid putting the pointcloud itself in global coordinates
    # by combining these transforms into one.
    lidar_to_crop = reduce(np.dot, [global_to_crop, ego_to_global, lidar_to_ego, map_align_transform])

    # Ego car parameters, width and length.
    front_length = 4.049
    rear_length = 1.127
    ego_car_length = front_length + rear_length
    ego_car_width = 1.1485 * 2.0

    ego_pose_np = np.array([ego_poses[0].x, ego_poses[0].y, ego_poses[0].z, 1])  # type: ignore
    ego_box = Box3D(
        center=(ego_pose_np[0], ego_pose_np[1], ego_pose_np[2]),
        size=(ego_car_width, ego_car_length, 1.78),
        orientation=ego_poses[0].quaternion,  # type: ignore
    )
    ego_box.transform(ego_poses[0].trans_matrix_inv)  # type: ignore
    ego_box.transform(map_align_transform)
    ego_box.transform(lidar_to_ego)
    ego_box.transform(ego_to_global)
    ego_box.scale(map_scale)
    ego_box.translate(map_translation)

    color = (1.0, 0.0, 0.0)  # Ego car in red color.
    colors: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], str]] = (color, color, "k")
    ego_box.render(ax, colors=colors)

    if render_future_ego_poses:
        ego_poses_np = get_future_ego_trajectory(
            lidarpc_rec=lidarpc_rec,
            future_ego_poses=ego_poses,
            transformmatrix=lidar_to_crop,
            future_horizon_len_s=6.0,
            future_interval_s=0.5,
        )
        draw_future_ego_poses(ego_box, ego_poses_np, color, ax)

    if render_vector_map:
        vector_map = lidarpc_rec.ego_pose.get_vector_map(maps_db=db.maps_db, xrange=xrange, yrange=yrange)  # type: ignore
        lane_coords = vector_map.coords  # [num_lanes, 2, 2]

        for coords in lane_coords:
            # converting 2D points to 3D to apply transformations.
            start = np.array([coords[0][0], coords[0][1], 0.0])  # type: ignore
            end = np.array([coords[1][0], coords[1][1], 0.0])  # type: ignore

            start = transform(start, map_align_transform)
            end = transform(end, map_align_transform)

            start = transform(start, lidar_to_ego)
            end = transform(end, lidar_to_ego)

            start = transform(start, ego_to_global)
            end = transform(end, ego_to_global)

            start = scale(start, map_scale)
            end = scale(end, map_scale)

            start = translate(start, map_translation)
            end = translate(end, map_translation)

            # converting back to 2D points.
            line = geometry.LineString([start[:-1], end[:-1]])
            xx, yy = line.coords.xy
            ax.plot(xx, yy, color="y", alpha=0.3)

    pointcloud.transform(lidar_to_crop)
    ax.scatter(pointcloud.points[0, :], pointcloud.points[1, :], c="g", s=1, alpha=0.2)

    if track_token is None and with_random_color:
        cmap = plt.cm.get_cmap("Dark2", len(boxes_lidar))

    for idx, box in enumerate(boxes_lidar):
        box_copy = box.copy()
        if track_token is not None:
            if box_copy.track_token != track_token:
                continue
        if (np.abs(box_copy.center[0]) <= radius) and (np.abs(box_copy.center[1]) <= radius):
            colors, marker = get_colors_marker(labelmap, box_copy)
            if track_token is None and with_random_color:
                c = np.array(cmap(idx)[:3])  # type: ignore
                colors = (c, c, "k")  # type: ignore

            box_copy.transform(map_align_transform)
            box_copy.transform(lidar_to_ego)
            box_copy.transform(ego_to_global)
            box_copy.scale(map_scale)
            box_copy.translate(map_translation)
            box_copy.render(ax, colors=colors, marker=marker, with_velocity=render_boxes_with_velocity)

    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()

    return ax


def boxes_lidar_to_img(
    lidar_record: LidarPc,  # type: ignore
    img_record: Image,  # type: ignore
    boxes_lidar: List[Box3D],
) -> List[Box3D]:
    """
    This function transforms the boxes in the Lidar frame to the image frame.
    :param lidar_record: The SampleData record for the point cloud.
    :param img_record: The SampleData record for the image.
    :param boxes_lidar: List of boxes in the Lidar frame (given by lidar_record).
    :return: List of boxes in the image frame (given by img_record).
    """
    cam_intrinsic = img_record.camera.intrinsic_np  # type: ignore
    imsize = (img_record.camera.width, img_record.camera.height)  # type: ignore

    ego_from_lidar = lidar_record.lidar.trans_matrix  # type: ignore
    global_from_ego = lidar_record.ego_pose.trans_matrix  # type: ignore
    ego_from_global = img_record.ego_pose.trans_matrix_inv  # type: ignore
    img_from_ego = img_record.camera.trans_matrix_inv  # type: ignore

    # Fuse four transformation matrices into one
    trans_matrix = reduce(np.dot, [img_from_ego, ego_from_global, global_from_ego, ego_from_lidar])

    boxes_img = []
    for box in boxes_lidar:

        box = box.copy()

        box.transform(trans_matrix)

        if box_in_image(box, cam_intrinsic, imsize):
            boxes_img.append(box)

    return boxes_img


def load_pointcloud_from_pc(
    nuplandb: NuPlanDB,  # type: ignore
    token: str,
    nsweeps: Union[int, List[int]],
    max_distance: float,
    min_distance: float,
    drivable_area: bool = False,
    map_dilation: float = 0.0,
    use_intensity: bool = True,
    use_ring: bool = False,
    use_lidar_index: bool = False,
    lidar_indices: Optional[Tuple[int, ...]] = None,
    sample_apillar_lidar_rings: bool = False,
    sweep_map: str = "time_lag",
) -> LidarPointCloud:
    """
    Loads one or more sweeps of a LIDAR pointcloud from the database using a SampleData record of NuPlanDB.
    :param nuplandb: The multimodal database used in this dataset.
    :param token: Token for the Lidar pointcloud.
    :param nsweeps: The number of past LIDAR sweeps used in the model.
        Alternatively, it is possible to provide a list of relative sweep indices, with:
        - Negative numbers corresponding to past sweeps.
        - 0 corresponding to the present sweep.
        - Positive numbers corresponding to future sweeps.
    :param max_distance: Radius outside which the points will be removed. Helps speed up caching and building the
        GT database.
    :param min_distance: Radius below which near points will be removed. This is usually recommended by the lidar
        manufacturer.
    :param drivable_area: Whether the pointcloud should be filtered based on drivable_area mask.
    :param map_dilation: Map dilation factor in meters.
    :param use_intensity: See prepare_pointcloud_points documentation for details.
    :param use_ring: See prepare_pointcloud_points documentation for details.
    :param use_lidar_index: Whether to use lidar index as a decoration.
    :param lidar_indices: See prepare_pointcloud_points documentation for details.
    :param sample_apillar_lidar_rings: Whether you want to sample rings for the A-pillar lidars.
    :param sweep_map: What to append to the lidar points to give information about what sweep it belongs to.
        Options: 'time_lag' and 'sweep_idx'.
    :return: The pointcloud.
    """
    # Check inputs
    assert sweep_map in ["time_lag", "sweep_idx"]
    if isinstance(nsweeps, int):
        nsweeps = list(range(-nsweeps + 1, 0 + 1))  # Use present sweep and past (nsweeps-1) sweeps
    elif isinstance(nsweeps, list):
        assert 0 in nsweeps, f"Error: Present sweep (0) must be included! nsweeps is: {nsweeps}"
    else:
        raise TypeError("Invalid nsweeps type: {}".format(type(nsweeps)))
    assert sorted(nsweeps) == nsweeps, "Error: nsweeps must be sorted in ascending order!"

    lidarpc_rec = nuplandb.lidar_pc[token]  # type: ignore
    time_current = lidarpc_rec.timestamp

    if len(nsweeps) > 1:
        # Homogeneous transformation matrix from lidar to ego car frame. Here we use the configs from
        # the current frame since, which (reasonably) assumes calibration hasn't changed
        car_from_lidar = lidarpc_rec.lidar.trans_matrix

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = lidarpc_rec.ego_pose.trans_matrix_inv

        # Homogeneous transform from ego car frame to lidar frame
        lidar_from_car = lidarpc_rec.lidar.trans_matrix_inv

    init = False
    for rel_sweep_idx, sweep_idx in enumerate(nsweeps):
        sweep_lidarpc_rec = _get_past_future_sweep(lidarpc_rec, sweep_idx)
        if sweep_lidarpc_rec is None:  # No previous or future sample data
            continue

        # Load up the pointcloud
        sweep_pc = sweep_lidarpc_rec.load(nuplandb)
        sweep_pc = prepare_pointcloud_points(
            sweep_pc,
            use_intensity=use_intensity,
            use_ring=use_ring,
            use_lidar_index=use_lidar_index,
            lidar_indices=lidar_indices,
            sample_apillar_lidar_rings=sample_apillar_lidar_rings,
        )

        # Remove points that are too close.
        # This is typically used to filter out points on the ego vehicle itself from each sweep.
        sweep_pc.remove_close(min_distance)

        # All but the present sweep are transformed to the present lidar coordinate frame
        if sweep_idx != 0:
            # Get sweep pose
            sweep_pose_rec = sweep_lidarpc_rec.ego_pose
            global_from_car = sweep_pose_rec.trans_matrix

            # Fuse four transformation matrices into one and perform transform
            # noinspection PyUnboundLocalVariable
            trans_matrix = reduce(np.dot, [lidar_from_car, car_from_global, global_from_car, car_from_lidar])
            sweep_pc.transform(trans_matrix)

        # Remove points that are too far *after* putting the points in the present lidar coordinate frame.
        # This is to ensure that the output pointcloud has no points further than
        # max_distance from the present lidar coordinate frame, even if those points came from
        # other sweeps.
        sweep_pc.radius_filter(max_distance)

        # Augment with sweep idx (Pixor) or time different (PointPillars)
        if sweep_map == "sweep_idx":
            rel_sweep_idx_pixor = np.array(rel_sweep_idx, dtype=np.float32) + 1
            assert rel_sweep_idx_pixor > 0  # Must be in [1, n] since pixor_cython uses unsigned ints
            sweep_vector = rel_sweep_idx_pixor * np.ones((1, sweep_pc.nbr_points()), dtype=np.float32)
        elif sweep_map == "time_lag":
            # Positive difference for past sweeps. Do not change this or existing models will be affected!
            # noinspection PyUnboundLocalVariable
            time_lag = time_current - sweep_lidarpc_rec.timestamp if sweep_idx != 0 else 0
            sweep_vector = 1e-6 * time_lag * np.ones((1, sweep_pc.nbr_points()), dtype=np.float32)
        else:
            raise ValueError("Cannot recognize sweep_map type: {}".format(sweep_map))

        sweep_pc.points = np.concatenate((sweep_pc.points, sweep_vector), axis=0)

        # Stack up the pointclouds.
        if not init:
            pc: LidarPointCloud = sweep_pc
            init = True
        else:
            # noinspection PyUnboundLocalVariable
            pc.points = np.hstack((pc.points, sweep_pc.points))

    # TODO: Revive the filtering once we have the map ready.
    # Filter points based on the drivable area mask.
    # if drivable_area:
    #    sp_map = lidarpc_rec.log.drivable_area
    #
    #    # First move points to ego vehicle coord system (from sensor coordinate system).
    #    # Then, move points to global coord system (from ego-vehicle coordinate system).
    #    # Then, move points to map coord system (from global coordinate system).
    #    # We fuse these into one "sensor to map coordinates" transform for efficiency.
    #    trans_matrix = reduce(np.dot, [
    #        sp_map.transform_matrix,
    #        lidarpc_rec.ego_pose.trans_matrix,
    #        lidarpc_rec.lidar.trans_matrix
    #    ])
    #
    #    # Modify the transform used by `sp_map.is_on_mask` to avoid transforming
    #    # the whole pointcloud multiple times. We want to avoid putting the
    #    # pointcloud itself in global coordinates, because UTM coordinates can
    #    # have Y values of over 4*10^6, and the pointcloud uses float32.
    #    # Modifying the internal variable of the MapLayer object is okay since
    #    # `sample.log.drivable_area` returns a new object every time
    #    # it's called.
    #    sp_map._transform_matrix = trans_matrix
    #
    #    # Find out points to keep based on drivable area.
    #    keep = sp_map.is_on_mask(pc.points[0], pc.points[1], dilation=map_dilation)
    #    if len(keep) > 0 and np.mean(keep) < 0.05:
    #        logger.warning('Only less than 5 percent points left after drivable_area filtering for {}.'.format(token))
    #
    #    # Filter the points.
    #    pc.points = pc.points[:, keep]
    #
    #    # An attempt to prevent dataloader from crashing. Otherwise, the dataset was filling up all the RAM on the
    #    # system.
    #    del keep, sp_map

    return pc


def _get_past_future_sweep(
    present_lidarpc: LidarPc, sweep_idx: int  # type: ignore
) -> LidarPc:  # type: ignore
    """
    Find a past or future sweep given the present sweep and its index.
    :param present_lidarpc: The present sweep.
    :param sweep_idx: The sweep index.
        - Negative numbers corresponding to past sweeps.
        - 0 corresponding to the present sweep.
        - Positive numbers corresponding to future sweeps.
    :returns: The specified sweep or None if we hit the start or end of an extraction.
    """
    # Initialize with present sweep. If sweep_idx == 0, we return it.
    cur_lidarpc = present_lidarpc
    for _ in range(abs(sweep_idx)):
        # Positive sweep index means future sweeps
        if sweep_idx > 0:
            if cur_lidarpc.next is None:  # type: ignore
                return None
            else:
                cur_lidarpc = cur_lidarpc.next  # type: ignore

        # Negative sweep index means past sweeps
        elif sweep_idx < 0:
            if cur_lidarpc.prev is None:  # type: ignore
                return None
            else:
                cur_lidarpc = cur_lidarpc.prev  # type: ignore

    return cur_lidarpc


def prepare_pointcloud_points(
    pc: LidarPointCloud,
    use_intensity: bool = True,
    use_ring: bool = False,
    use_lidar_index: bool = False,
    lidar_indices: Optional[Tuple[int, ...]] = None,
    sample_apillar_lidar_rings: bool = False,
) -> LidarPointCloud:
    """
    Prepare the lidar points.
    There are two independent steps:
        - filter points to only use a subset of the lidars
        - change the decorations (intensity and ring)
    :param pc: Pointcloud input.
    :param use_intensity: Whether to use intensity or not.
    :param use_ring: Whether to use ring index or not.
    :param use_lidar_index: Whether to use lidar index as a decoration.
    :param lidar_indices: Which lidars to keep.
        MergedPointCloud has following options:
            0: top lidar
            1: right A pillar lidar
            2: left A pillar lidar
            3: back lidar
            4: front lidar
            None: Use all lidars
    :param sample_apillar_lidar_rings: Whether you want to sample rings for the A-pillar lidars.
    :return: Modified pointcloud.
    """
    a_pillar_lidar_indices = (1, 2)

    ring_indices_to_keep = [0, 1, 2, 3, 4, 5, 6, 8, 11, 17, 23, 29, 35, 38, 39]

    if lidar_indices is None:
        # Use all 5 lidars.
        if sample_apillar_lidar_rings:
            # The two A-Pillar Lidars will be downsampled while all the points from the other three lidars are used.
            keep = np.zeros(pc.points.shape[1])

            # First select all points for non a-pillar lidars.
            keep = np.logical_or(
                keep, (pc.points[5] != a_pillar_lidar_indices[0]) & (pc.points[5] != a_pillar_lidar_indices[1])
            )

            # Only select specific rings from the a-pillar lidars now.
            for index in a_pillar_lidar_indices:
                keep = np.logical_or(keep, (pc.points[5] == index) & np.isin(pc.points[4], ring_indices_to_keep))

            pc.points = pc.points[:, keep]
    else:
        # Use a subset of the MergedPointCloud
        keep = np.zeros(pc.points.shape[1])
        for index in lidar_indices:
            if sample_apillar_lidar_rings and index in a_pillar_lidar_indices:
                current_keep = (pc.points[5] == index) & np.isin(pc.points[4], ring_indices_to_keep)
            else:
                current_keep = pc.points[5] == index
            keep = np.logical_or(keep, current_keep)

        pc.points = pc.points[:, keep]

    # Which information should we keep?
    decoration_index = [0, 1, 2]  # Always use x, y, z.
    if use_intensity:
        decoration_index += [3]
    if use_ring:
        decoration_index += [4]
    if use_lidar_index:
        decoration_index += [5]

    # Filter the points
    pc.points = pc.points[np.array(decoration_index)]

    return pc


def load_boxes_from_lidarpc(
    nuplandb: NuPlanDB,  # type: ignore
    lidarpc_rec: LidarPc,  # type: ignore
    target_category_names: List[str],
    filter_boxes: bool,
    max_distance: float,
    future_horizon_len_s: float = 0.0,
    future_interval_s: float = 0.5,
    category2id: Optional[Dict[str, int]] = None,
    map_dilation: float = 0.0,
) -> Dict[str, List[Box3D]]:
    """
    Load all the boxes for a LidarPc.
    :param nuplandb: The multimodal database used in this dataset.
    :param lidarpc_rec: Lidar sample record.
    :param target_category_names: Global names corresponding to the boxes we are interested in obtaining.
    :param filter_boxes: Whether to filter the boxes to be on the drivable area + dilation factor.
    :param max_distance: Radius outside which the boxes will be removed. Helps speed up caching and building the
        GT database.
    :param future_horizon_len_s: Num seconds in the future where we want a future box.
        If a value is provided, the center coordinates and orientation for each box will be provided at 0.5 sec
        intervals. If the value is 0 (default), the function will not provide future center coordinates or orientation.
    :param future_interval_s: Time interval between future waypoints in seconds.
    :param category2id: Mapping from category name to id. This parameter is optional and if provided, it is used to
        populate the box.label property when applicable.
    :param map_dilation: Map dilation factor in meters.
    :return: Dictionary mapping global names of desired categories to list of corresponding boxes.
    """
    # Get all the GT boxes in global frame for that sample
    if future_horizon_len_s:
        assert 0 < future_interval_s <= future_horizon_len_s
        all_boxes = lidarpc_rec.boxes_with_future_waypoints(  # type: ignore
            future_horizon_len_s=future_horizon_len_s, future_interval_s=future_interval_s
        )
    else:
        all_boxes = lidarpc_rec.boxes()  # type: ignore

    # Filter boxes for relevant classes and bikeracks.
    global2boxes: Dict[str, List[Box3D]] = {global_name: [] for global_name in target_category_names}
    for box in all_boxes:
        current_global_name = nuplandb.lidar_box[box.token].category.name  # type: ignore
        if current_global_name in target_category_names:
            if category2id and current_global_name in list(category2id.keys()):
                box.label = category2id[current_global_name]
            global2boxes[current_global_name].append(box)

    # TODO: Revive the filtering once we have the map ready.
    # Filter the boxes based on the drivable area mask.
    # if filter_boxes:
    #    # Get the drivable area mask
    #    sp_map = lidarpc_rec.log.drivable_area
    #
    #    for global_name, boxes in global2boxes.items():
    #        filtered_boxes = [
    #            box for box in boxes if sp_map.is_on_mask(box.center[0], box.center[1], dilation=map_dilation)
    #        ]
    #        global2boxes[global_name] = filtered_boxes
    #
    #    # An attempt to prevent dataloader from crashing. Otherwise, the dataset was filling up all the RAM on the
    #    # system.
    #    del sp_map

    # Transform the boxes to the sensor coordinate frame
    for global_name, boxes in global2boxes.items():
        car_from_global = lidarpc_rec.ego_pose.trans_matrix_inv  # type: ignore
        lidar_from_car = lidarpc_rec.lidar.trans_matrix_inv  # type: ignore
        trans_matrix = reduce(np.dot, [lidar_from_car, car_from_global])

        transformed_boxes = [_box_transform(box, trans_matrix) for box in boxes]
        filtered_boxes = [box for box in transformed_boxes if box.distance_plane < max_distance]
        global2boxes[global_name] = filtered_boxes

    return global2boxes


def _box_transform(box: Box3D, trans_matrix: npt.NDArray[np.float64]) -> Box3D:
    """
    Helper method so box transform can be done in a list comprehension.
    :param box: Box to transform.
    :param trans_matrix: <np.float: 4, 4> Transformation matrix.
    :return: Transformed box.
    """
    box.transform(trans_matrix)
    return box


def split_vehicles_by_size(boxes: List[Box3D], short_vehicles_id: int, long_vehicles_id: int) -> List[Box3D]:
    """
    This function splits vehicles into a different classes based on whether they are longer or shorter than 7m
    in length. It assigns labels to boxes
    :param boxes: List of boxes containing annotations of all types.
    :param short_vehicles_id: ID used in labelmap for short vehicles.
    :param long_vehicles_id: ID used in labelmap for long vehicles.
    :return: List of boxes where vehicles have been split into two classes and labeled accordingly.
    """
    final_boxes = []

    # First get all boxes. Apply the label for short or long vehicles accordingly.
    # All the remaining boxes will be added to the list final_boxes.
    for box in boxes:
        if box.label == short_vehicles_id or box.label == long_vehicles_id:
            if box.length <= 7:
                box.label = short_vehicles_id
                final_boxes.append(box)
            if box.length > 7:
                box.label = long_vehicles_id
                final_boxes.append(box)

        else:
            # For other classes, there is no change.
            final_boxes.append(box)

    return final_boxes


def crop_rect(
    img: npt.NDArray[np.float64], rect: Tuple[Tuple[float, float], Tuple[float, float], float]
) -> npt.NDArray[np.float64]:
    """
    Crop a rectangle from a 2D image.
    :param img: Numpy array containing the image.
    :param rect: Rectangle defined by cv2.minAreaRect.
    :return: Cropped img.
    """
    fcenter = rect[0]
    fsize = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, fcenter)), tuple(map(int, fsize))

    # The simplest thing to do is rotate the whole image, then crop. However,
    # OpenCV2 gives an error if you try to warpAffine on images that have a
    # height or width greater than 32767, which is the case for some of our
    # larger maps that get passed to this function. To work around this, we take
    # a crop of the original image, called warpAffine on that, then take the
    # final crop.
    rect_diagonal_length = math.ceil(math.sqrt(size[0] ** 2 + size[1] ** 2))
    crop_size_before_rotate = (rect_diagonal_length, rect_diagonal_length)
    crop_before_rotate = cv2.getRectSubPix(img, crop_size_before_rotate, center)

    new_center = (rect_diagonal_length / 2, rect_diagonal_length / 2)
    M = cv2.getRotationMatrix2D(new_center, angle, 1)
    rotated_img: npt.NDArray[np.float64] = cv2.warpAffine(crop_before_rotate, M, crop_size_before_rotate)
    cropped_img: npt.NDArray[np.float64] = cv2.getRectSubPix(rotated_img, size, new_center)

    return cropped_img


def project_lidarpcs_to_camera(
    pc: LidarPointCloud,
    transform: npt.NDArray[np.float64],
    camera_intrinsic: npt.NDArray[np.float64],
    width: int,
    height: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool8]]:
    """
    Project lidar pcs to a camera and return pcs with coordinate in a camera view.
    :param pc: Lidar point clouds.
    :param transform: <4, 4>. Matrix to transform point clouds to a camera view.
    :param camera_intrinsic: <3, 3>. Intrinsic matrix of a camera.
    :param width: Image width.
    :param height: Image height.
    :return points: <np.float: 3, number of points>. Point cloud with their coordinates in a camera.
            masks: <np.bool: number of points>. A 1d-array of boolean to indicate which points are available in
            the camera.
    """
    # Transform pc to img.
    pc.transform(transform)

    # Take the depth after transformation.
    depths = pc.points[2, :]

    # Take the actual picture (matrix multiplication with camera - matrix + renormalization).
    points = view_points(pc.points[:3, :], camera_intrinsic, normalize=True)

    # Finally filter away points outside the image.
    mask = np.ones(pc.points.shape[1], dtype=bool)  # type: ignore

    # Remove points from the back of the ego vehicle.
    mask = np.logical_and(mask, depths > 0.0)

    # Remove points outside the camera image in the image x channel.
    mask = np.logical_and(mask, points[0, :] > 0)
    mask = np.logical_and(mask, points[0, :] < width - 1)

    # Remove points outside the camera image in the image y channel.
    mask = np.logical_and(mask, points[1, :] > 0)
    mask = np.logical_and(mask, points[1, :] < height - 1)

    points = points[:, mask]

    # Get the original depth info after the transformation.
    points[2, :] = depths[mask]

    return points, mask
