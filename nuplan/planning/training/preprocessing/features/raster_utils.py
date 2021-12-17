from __future__ import annotations

from array import array
from typing import Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import BaselinePath, PolygonMapObject
from nuplan.planning.simulation.observation.observation_type import Detections


def _linestring_to_coords(geometry: List[BaselinePath]) -> List[Tuple[array]]:  # type: ignore
    """
    Get 2d coordinates of the endpoints of line segment string.
    The line segment string is a shapely.geometry.linestring.
    :param geometry: the line segment string.
    :return: 2d coordinates of the endpoints of line segment string.
    """
    return [element.baseline_path().linestring.coords.xy for element in geometry]


def _polygon_to_coords(geometry: List[PolygonMapObject]) -> List[Tuple[array]]:  # type: ignore
    """
    Get 2d coordinates of the vertices of a polygon.
    The polygon is a shapely.geometry.polygon.
    :param geometry: the polygon.
    :return: 2d coordinates of the vertices of the polygon.
    """
    return [element.polygon.exterior.coords.xy for element in geometry]


def _cartesian_to_projective_coords(coords: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Convert from cartesian coordinates to projective coordinates.
    :param coords: the 2d coordinates of shape (N, 2) where N is the number of points.
    :return: the resulting projective coordinates of shape (N, 3).
    """
    return np.pad(coords, ((0, 0), (0, 1)), 'constant', constant_values=1.0)  # type: ignore


def _get_layer_coords(
    ego_pose: EgoState,
    map_api: AbstractMap,
    map_layer_name: SemanticMapLayer,
    map_layer_geometry: str,
    radius: float,
) -> List[npt.NDArray[np.float32]]:
    """
    Constructs the map layer of the raster by converting vector map to raster map.

    :param ego_state: SE2 state of ego.
    :param map_api: map api
    :param map_layer_name: name of the vector map layer to create a raster from.
    :param map_layer_geometry: geometric primitive of the vector map layer. i.e. either polygon or linestring.
    :param radius: [m] the radius of the square raster map.
    :return: the list of 2d coordinates which represent the shape of the map.
    """

    ego_position = Point2D(ego_pose.rear_axle.x, ego_pose.rear_axle.y)
    nearest_vector_map = map_api.get_proximal_map_objects(
        layers=[map_layer_name],
        point=ego_position,
        radius=radius,
    )
    geometry = nearest_vector_map[map_layer_name]

    if len(geometry):
        global_transform = np.linalg.inv(ego_pose.rear_axle.as_matrix())  # type: ignore

        # By default the map is right-oriented, this makes it top-oriented.
        map_align_transform = R.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
        transform = map_align_transform @ global_transform

        if map_layer_geometry == 'polygon':
            object_coords = _polygon_to_coords(geometry)
        elif map_layer_geometry == 'linestring':
            object_coords = _linestring_to_coords(geometry)
        else:
            raise RuntimeError(f'Layer geometry {map_layer_geometry} type not supported')

        object_coords = [np.vstack(coords).T for coords in object_coords]  # type: ignore
        object_coords = [(transform @ _cartesian_to_projective_coords(coords).T).T[:, :2]  # type: ignore
                         for coords in object_coords]
    else:
        object_coords = []

    return object_coords  # type: ignore


def _draw_polygon_image(
    image: npt.NDArray[np.uint8],
    object_coords: List[npt.NDArray[np.float32]],
    radius: float,
    resolution: float,
    color: float,
    bit_shift: int = 12,
) -> npt.NDArray[np.uint8]:
    """
    Draw a map feature consisting of polygons using a list of its coordinates.
    :param image: the raster map on which the map feature will be drawn
    :param object_coords: the coordinates that represents the shape of the map feature.
    :param radius: the radius of the square raster map.
    :param resolution: [m] pixel size in meters.
    :param color: color of the map feature.
    :param bit_shift: bit shift of the polygon used in opencv.
    :return: the resulting raster map with the map feature.
    """

    if len(object_coords):
        for coords in object_coords:
            index_coords = (radius + coords) / resolution
            shifted_index_coords = (index_coords * 2 ** bit_shift).astype(np.int64)
            cv2.fillPoly(image, shifted_index_coords[None], color=color, shift=bit_shift, lineType=cv2.LINE_AA)

    return image


def _draw_linestring_image(
    image: npt.NDArray[np.uint8],
    object_coords: List[npt.NDArray[np.float32]],
    radius: float,
    resolution: float,
    baseline_path_thickness: int,
    color: float = 1.0,
    bit_shift: int = 13,
) -> npt.NDArray[np.uint8]:
    """
    Draw a map feature consisting of linestring using a list of its coordinates.
    :param image: the raster map on which the map feature will be drawn
    :param object_coords: the coordinates that represents the shape of the map feature.
    :param radius: the radius of the square raster map.
    :param resoluton: [m] pixel size in meters.
    :param color: color of the map feature.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :param bit_shift: bit shift of the polylines used in opencv.
    :return: the resulting raster map with the map feature.
    """

    if len(object_coords):
        for coords in object_coords:
            index_coords = (radius + coords) / resolution
            shifted_index_coords = (index_coords * 2 ** bit_shift).astype(np.int64)
            cv2.polylines(
                image, [shifted_index_coords], isClosed=False, color=color, thickness=baseline_path_thickness,
                shift=bit_shift, lineType=cv2.LINE_AA)

    return image


def get_roadmap_raster(
    ego_state: EgoState,
    map_api: AbstractMap,
    map_features: Dict[str, int],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    resolution: float,
) -> npt.NDArray[np.float32]:
    """
    Constructs the map layer of the raster by converting vector map to raster map.

    :param ego_state: SE2 state of ego.
    :param map_api: map api.
    :param map_features: name of map features to be drawn and its color for encoding.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :return roadmap_raster: the constructed map raster layer.
    """
    # Assume the raster has a square shape.
    assert (x_range[1] - x_range[0]) == (y_range[1] - y_range[0]), \
        f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'

    radius = (x_range[1] - x_range[0]) / 2
    roadmap_raster = np.zeros(raster_shape, dtype=np.float32)

    for feature_name, feature_color in map_features.items():
        coords = _get_layer_coords(ego_state, map_api, SemanticMapLayer[feature_name], 'polygon', radius)
        roadmap_raster = _draw_polygon_image(roadmap_raster, coords, radius, resolution, feature_color)

    # Flip the agents_raster along the horizontal axis.
    roadmap_raster = np.flip(roadmap_raster, axis=0)  # type: ignore
    roadmap_raster = np.ascontiguousarray(roadmap_raster, dtype=np.float32)
    return roadmap_raster


def get_agents_raster(
    ego_state: EgoState,
    detections: Detections,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    polygon_bit_shift: int = 9,
) -> npt.NDArray[np.float32]:
    """
    Constructs the agents layer of the raster by transforming all detected boxes around the agent
    and creating polygons of them in a raster grid.

    :param ego_state: SE2 state of ego.
    :param detections: list of 3D bounding box of detected agents.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param polygon_bit_shift: bit shift of the polygon used in opencv.
    :return: constructed agents raster layer.
    """

    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape

    agents_raster = np.zeros(raster_shape, dtype=np.float32)

    ego_to_global = ego_state.rear_axle.as_matrix_3d()
    global_to_ego = np.linalg.inv(ego_to_global)  # type: ignore

    # Extract the rotation matrix from the transformation matrix which is in the projective coordinate.
    map_align_rot = R.from_matrix(ego_to_global[:3, :3].T)
    map_align_rot_angle = map_align_rot.as_euler('zxy')[0] + (np.pi / 2)
    map_align_transform = Quaternion(axis=[0, 0, 1], angle=map_align_rot_angle).transformation_matrix
    transforms = ego_to_global @ map_align_transform @ global_to_ego

    # Ego translation respect to global frame
    ego_translation = -np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, 0])

    # Retrieve the scenario's boxes.
    boxes = [box.copy() for box in detections.boxes]

    for box in boxes:
        # Transform the agent bounding boxes to ego coordinate and rotate them to make the ego face up.
        box.transform(transforms)
        box.translate(ego_translation)

        # Filter out boxes outside the raster.
        valid_x = x_range[0] < box.center[0] < x_range[1]
        valid_y = y_range[0] < box.center[1] < y_range[1]
        if not (valid_x and valid_y):
            continue

        # Get the 2D coordinate of the detected agents.
        box_bottom_corners = box.bottom_corners[:2, :]
        x_corners, y_corners = box_bottom_corners

        # Discretize
        y_corners = (y_corners - ymin) / (ymax - ymin) * height
        x_corners = (x_corners - xmin) / (xmax - xmin) * width

        box_2d_coords = np.stack([x_corners, y_corners], axis=1)
        box_2d_coords = np.expand_dims(box_2d_coords, axis=0)  # type: ignore

        # Draw the box as a filled polygon on the raster layer.
        box_2d_coords = (box_2d_coords * 2 ** polygon_bit_shift).astype(np.int32)
        cv2.fillPoly(agents_raster, box_2d_coords, color=1.0, shift=polygon_bit_shift, lineType=cv2.LINE_AA)

    # Flip the agents_raster along the horizontal axis.
    agents_raster = np.asarray(agents_raster)
    agents_raster = np.flip(agents_raster, axis=0)  # type: ignore
    agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)

    return agents_raster


def get_ego_raster(
    raster_shape: Tuple[int, int],
    ego_longitudinal_offset: float,
    ego_width_pixels: float,
    ego_front_length_pixels: float,
    ego_rear_length_pixels: float,
) -> npt.NDArray[np.float32]:
    """
    Constructs the ego layer of the raster by drawing a polygon of the ego's extent in the middle of the grid.
    :param raster_shape: shape of the target raster.
    :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
    :param ego_width_pixels: width of the ego vehicle in pixels.
    :param ego_front_length_pixels: distance between the rear axle and the front bumper in pixels.
    :param ego_rear_length_pixels: distance between the rear axle and the rear bumper in pixels.
    :return: constructed ego raster layer.
    """
    ego_raster = np.zeros(raster_shape, dtype=np.float32)

    # Construct a rectangle representing the ego vehicle in the center of the raster.
    map_x_center = int(raster_shape[1] * 0.5)
    map_y_center = int(raster_shape[0] * (0.5 + ego_longitudinal_offset))
    ego_top_left = (map_x_center - ego_width_pixels // 2, map_y_center - ego_front_length_pixels)
    ego_bottom_right = (map_x_center + ego_width_pixels // 2, map_y_center + ego_rear_length_pixels)
    cv2.rectangle(ego_raster, ego_top_left, ego_bottom_right, 1, -1)

    return np.asarray(ego_raster)


def get_baseline_paths_raster(
    ego_state: EgoState,
    map_api: AbstractMap,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    resolution: float,
    baseline_path_thickness: int = 1,
) -> npt.NDArray[np.float32]:
    """
    Constructs the baseline paths layer by converting vector map to raster map.

    :param ego_state: SE2 state of ego.
    :param map_api: map api
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :return baseline_paths_raster: the constructed baseline paths layer.
    """

    # Assume the raster has a square shape.
    if (x_range[1] - x_range[0]) != (y_range[1] - y_range[0]):
        raise ValueError(f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}')

    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster = np.zeros(raster_shape, dtype=np.float32)

    for map_features in ['LANE', 'LANE_CONNECTOR']:
        baseline_paths_coords = _get_layer_coords(
            ego_state, map_api, SemanticMapLayer[map_features], 'linestring', radius)
        baseline_paths_raster = _draw_linestring_image(
            baseline_paths_raster, baseline_paths_coords, radius, resolution, baseline_path_thickness)

    # Flip the agents_raster along the horizontal axis.
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)  # type: ignore
    baseline_paths_raster = np.ascontiguousarray(baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster
