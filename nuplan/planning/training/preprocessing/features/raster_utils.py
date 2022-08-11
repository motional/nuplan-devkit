from __future__ import annotations

from array import array
from copy import deepcopy
from typing import Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import PolygonMapObject, PolylineMapObject
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

# TODO: Move traffic light color configurations to hydra yaml files.
# Color dict in the format of taffic_light_type: color tuple
BASELINE_TL_COLOR = {
    TrafficLightStatusType.RED: (1, 0, 0),
    TrafficLightStatusType.YELLOW: (1, 1, 0),
    TrafficLightStatusType.GREEN: (0, 1, 0),
    TrafficLightStatusType.UNKNOWN: (0, 0, 1),  # Also the deafult color for baseline path
}


def _linestring_to_coords(geometry: List[PolylineMapObject]) -> List[Tuple[array[float]]]:
    """
    Get 2d coordinates of the endpoints of line segment string.
    The line segment string is a shapely.geometry.linestring.
    :param geometry: the line segment string.
    :return: 2d coordinates of the endpoints of line segment string.
    """
    return [element.baseline_path.linestring.coords.xy for element in geometry]


def _polygon_to_coords(geometry: List[PolygonMapObject]) -> List[Tuple[array[float]]]:
    """
    Get 2d coordinates of the vertices of a polygon.
    The polygon is a shapely.geometry.polygon.
    :param geometry: the polygon.
    :return: 2d coordinates of the vertices of the polygon.
    """
    return [element.polygon.exterior.coords.xy for element in geometry]


def _cartesian_to_projective_coords(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert from cartesian coordinates to projective coordinates.
    :param coords: the 2d coordinates of shape (N, 2) where N is the number of points.
    :return: the resulting projective coordinates of shape (N, 3).
    """
    return np.pad(coords, ((0, 0), (0, 1)), 'constant', constant_values=1.0)


def _get_layer_coords(
    agent: AgentState,
    map_api: AbstractMap,
    map_layer_name: SemanticMapLayer,
    map_layer_geometry: str,
    radius: float,
) -> Tuple[List[npt.NDArray[np.float64]], List[str]]:
    """
    Construct the map layer of the raster by converting vector map to raster map, based on the focus agent.
    :param agent: the focus agent used for raster generating.
    :param map_api: map api
    :param map_layer_name: name of the vector map layer to create a raster from.
    :param map_layer_geometry: geometric primitive of the vector map layer. i.e. either polygon or linestring.
    :param radius: [m] the radius of the square raster map.
    :return
        object_coords: the list of 2d coordinates which represent the shape of the map.
        lane_ids: the list of ids for the map objects.
    """
    ego_position = Point2D(agent.center.x, agent.center.y)
    nearest_vector_map = map_api.get_proximal_map_objects(
        layers=[map_layer_name],
        point=ego_position,
        radius=radius,
    )
    geometry = nearest_vector_map[map_layer_name]

    if len(geometry):
        global_transform = np.linalg.inv(agent.center.as_matrix())

        # By default the map is right-oriented, this makes it top-oriented.
        map_align_transform = R.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
        transform = map_align_transform @ global_transform

        if map_layer_geometry == 'polygon':
            _object_coords = _polygon_to_coords(geometry)
        elif map_layer_geometry == 'linestring':
            _object_coords = _linestring_to_coords(geometry)
        else:
            raise RuntimeError(f'Layer geometry {map_layer_geometry} type not supported')

        object_coords: List[npt.NDArray[np.float64]] = [np.vstack(coords).T for coords in _object_coords]
        object_coords = [(transform @ _cartesian_to_projective_coords(coords).T).T[:, :2] for coords in object_coords]

        lane_ids = [lane.id for lane in geometry]
    else:
        object_coords = []
        lane_ids = []

    return object_coords, lane_ids


def _draw_polygon_image(
    image: npt.NDArray[np.float32],
    object_coords: List[npt.NDArray[np.float64]],
    radius: float,
    resolution: float,
    color: float,
    bit_shift: int = 12,
) -> npt.NDArray[np.float32]:
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
            shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)
            cv2.fillPoly(image, shifted_index_coords[None], color=color, shift=bit_shift, lineType=cv2.LINE_AA)

    return image


def _draw_linestring_image(
    image: npt.NDArray[np.float32],
    object_coords: List[npt.NDArray[np.float64]],
    radius: float,
    resolution: float,
    baseline_path_thickness: int,
    lane_colors: npt.NDArray[np.uint8],
    bit_shift: int = 13,
) -> npt.NDArray[np.float32]:
    """
    Draw a map feature consisting of linestring using a list of its coordinates.
    :param image: the raster map on which the map feature will be drawn
    :param object_coords: the coordinates that represents the shape of the map feature.
    :param radius: the radius of the square raster map.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :param lane_colors: an array indicate colors for each element of object_coords.
    :param bit_shift: bit shift of the polylines used in opencv.
    :return: the resulting raster map with the map feature.
    """
    if len(object_coords):
        assert len(object_coords) == len(lane_colors)
        for coords, lane_color in zip(object_coords, lane_colors):
            index_coords = (radius + coords) / resolution
            shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)
            # Add int() before lane_color to address the cv2 error: color should be numeric
            lane_color = int(lane_color) if np.isscalar(lane_color) else [int(item) for item in lane_color]  # type: ignore
            cv2.polylines(
                image,
                [shifted_index_coords],
                isClosed=False,
                color=lane_color,
                thickness=baseline_path_thickness,
                shift=bit_shift,
                lineType=cv2.LINE_AA,
            )

    return image


def get_roadmap_raster(
    focus_agent: AgentState,
    map_api: AbstractMap,
    map_features: Dict[str, int],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    resolution: float,
) -> npt.NDArray[np.float32]:
    """
    Construct the map layer of the raster by converting vector map to raster map.
    :param focus_agent: agent state representing ego.
    :param map_api: map api.
    :param map_features: name of map features to be drawn and its color for encoding.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :return roadmap_raster: the constructed map raster layer.
    """
    # Assume the raster has a square shape.
    assert (x_range[1] - x_range[0]) == (
        y_range[1] - y_range[0]
    ), f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'

    radius = (x_range[1] - x_range[0]) / 2
    roadmap_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)

    for feature_name, feature_color in map_features.items():
        coords, _ = _get_layer_coords(focus_agent, map_api, SemanticMapLayer[feature_name], 'polygon', radius)
        roadmap_raster = _draw_polygon_image(roadmap_raster, coords, radius, resolution, feature_color)

    # Flip the agents_raster along the horizontal axis.
    roadmap_raster = np.flip(roadmap_raster, axis=0)
    roadmap_raster = np.ascontiguousarray(roadmap_raster, dtype=np.float32)

    return roadmap_raster


def get_agents_raster(
    ego_state: EgoState,
    detections: DetectionsTracks,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    polygon_bit_shift: int = 9,
) -> npt.NDArray[np.float32]:
    """
    Construct the agents layer of the raster by transforming all detected boxes around the agent
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

    agents_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)

    ego_to_global = ego_state.rear_axle.as_matrix()
    global_to_ego = np.linalg.inv(ego_to_global)
    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()

    # Retrieve the scenario's boxes.
    tracked_objects = [deepcopy(tracked_object) for tracked_object in detections.tracked_objects]

    for tracked_object in tracked_objects:
        # Transform the box relative to agent.
        raster_object_matrix = north_aligned_transform @ global_to_ego @ tracked_object.center.as_matrix()
        raster_object_pose = StateSE2.from_matrix(raster_object_matrix)
        # Filter out boxes outside the raster.
        valid_x = x_range[0] < raster_object_pose.x < x_range[1]
        valid_y = y_range[0] < raster_object_pose.y < y_range[1]
        if not (valid_x and valid_y):
            continue

        # Get the 2D coordinate of the detected agents.
        raster_oriented_box = OrientedBox(
            raster_object_pose, tracked_object.box.length, tracked_object.box.width, tracked_object.box.height
        )
        box_bottom_corners = raster_oriented_box.all_corners()
        x_corners = np.asarray([corner.x for corner in box_bottom_corners])  # type: ignore
        y_corners = np.asarray([corner.y for corner in box_bottom_corners])  # type: ignore

        # Discretize
        y_corners = (y_corners - ymin) / (ymax - ymin) * height  # type: ignore
        x_corners = (x_corners - xmin) / (xmax - xmin) * width  # type: ignore

        box_2d_coords = np.stack([x_corners, y_corners], axis=1)  # type: ignore
        box_2d_coords = np.expand_dims(box_2d_coords, axis=0)

        # Draw the box as a filled polygon on the raster layer.
        box_2d_coords = (box_2d_coords * 2**polygon_bit_shift).astype(np.int32)
        cv2.fillPoly(agents_raster, box_2d_coords, color=1.0, shift=polygon_bit_shift, lineType=cv2.LINE_AA)

    # Flip the agents_raster along the horizontal axis.
    agents_raster = np.asarray(agents_raster)
    agents_raster = np.flip(agents_raster, axis=0)
    agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)

    return agents_raster


def get_focus_agent_raster(
    agent: AgentState,
    raster_shape: Tuple[int, int],
    ego_longitudinal_offset: float,
    target_pixel_size: float,
) -> npt.NDArray[np.float32]:
    """
    Construct the focus agent layer of the raster by drawing a polygon of the ego's extent in the middle of the grid.
    :param agent: Focus agent of the target raster.
    :param raster_shape: Shape of the target raster.
    :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
    :param target_pixel_size: [m] target pixel size in meters.
    :return: Constructed ego raster layer.
    """
    ego_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)

    # Construct a rectangle representing the ego vehicle in the center of the raster.
    map_x_center = int(raster_shape[1] * 0.5)
    map_y_center = int(raster_shape[0] * (0.5 + ego_longitudinal_offset))
    ego_width_pixels = int(agent.box.width / target_pixel_size)
    ego_length_pixels = int(agent.box.length / target_pixel_size)
    ego_top_left = (map_x_center - ego_width_pixels // 2, map_y_center - ego_length_pixels // 2)
    ego_bottom_right = (map_x_center + ego_width_pixels // 2, map_y_center + ego_length_pixels // 2)
    cv2.rectangle(ego_raster, ego_top_left, ego_bottom_right, 1, -1)

    return np.asarray(ego_raster)


def get_non_focus_agents_raster(
    focus_agent: AgentState,
    other_agents: List[Agent],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    polygon_bit_shift: int = 9,
) -> npt.NDArray[np.float32]:
    """
    Construct the agents layer of the raster by transforming all other agents around the focus agent
    and creating polygons of them in a raster grid.
    :param focus_agent: focus agent used for rasterization.
    :param other agents: list of agents including the ego AV but excluding the focus agent.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param polygon_bit_shift: bit shift of the polygon used in opencv.
    :return: constructed agents raster layer.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape

    agents_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)

    ego_to_global = focus_agent.center.as_matrix()
    global_to_ego = np.linalg.inv(ego_to_global)
    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()

    for tracked_object in other_agents:
        raster_object_matrix = north_aligned_transform @ global_to_ego @ tracked_object.center.as_matrix()
        raster_object_pose = StateSE2.from_matrix(raster_object_matrix)
        # Filter out boxes outside the raster.
        valid_x = x_range[0] < raster_object_pose.x < x_range[1]
        valid_y = y_range[0] < raster_object_pose.y < y_range[1]
        if not (valid_x and valid_y):
            continue

        # Get the 2D coordinate of the detected agents.
        raster_oriented_box = OrientedBox(
            raster_object_pose, tracked_object.box.length, tracked_object.box.width, tracked_object.box.height
        )
        box_bottom_corners = raster_oriented_box.all_corners()
        x_corners = np.asarray([corner.x for corner in box_bottom_corners])  # type: ignore
        y_corners = np.asarray([corner.y for corner in box_bottom_corners])  # type: ignore

        # Discretize
        y_corners = (y_corners - ymin) / (ymax - ymin) * height  # type: ignore
        x_corners = (x_corners - xmin) / (xmax - xmin) * width  # type: ignore

        box_2d_coords = np.stack([x_corners, y_corners], axis=1)  # type: ignore
        box_2d_coords = np.expand_dims(box_2d_coords, axis=0)

        # Draw the box as a filled polygon on the raster layer.
        box_2d_coords = (box_2d_coords * 2**polygon_bit_shift).astype(np.int32)
        cv2.fillPoly(agents_raster, box_2d_coords, color=1.0, shift=polygon_bit_shift, lineType=cv2.LINE_AA)

    # Flip the agents_raster along the horizontal axis.
    agents_raster = np.asarray(agents_raster)
    agents_raster = np.flip(agents_raster, axis=0)
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
    Construct the ego layer of the raster by drawing a polygon of the ego's extent in the middle of the grid.
    :param raster_shape: shape of the target raster.
    :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
    :param ego_width_pixels: width of the ego vehicle in pixels.
    :param ego_front_length_pixels: distance between the rear axle and the front bumper in pixels.
    :param ego_rear_length_pixels: distance between the rear axle and the rear bumper in pixels.
    :return: constructed ego raster layer.
    """
    ego_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)

    # Construct a rectangle representing the ego vehicle in the center of the raster.
    map_x_center = int(raster_shape[1] * 0.5)
    map_y_center = int(raster_shape[0] * (0.5 + ego_longitudinal_offset))
    ego_top_left = (map_x_center - ego_width_pixels // 2, map_y_center - ego_front_length_pixels)
    ego_bottom_right = (map_x_center + ego_width_pixels // 2, map_y_center + ego_rear_length_pixels)
    cv2.rectangle(ego_raster, ego_top_left, ego_bottom_right, 1, -1)

    return np.asarray(ego_raster)


def get_baseline_paths_raster(
    focus_agent: AgentState,
    map_api: AbstractMap,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    resolution: float,
    baseline_path_thickness: int = 1,
) -> npt.NDArray[np.float32]:
    """
    Construct the baseline paths layer by converting vector map to raster map.
    This funciton is for ego raster model, the baselin path only has one channel.
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
        raise ValueError(
            f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'
        )

    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)

    for map_features in ['LANE', 'LANE_CONNECTOR']:
        baseline_paths_coords, lane_ids = _get_layer_coords(
            agent=focus_agent,
            map_api=map_api,
            map_layer_name=SemanticMapLayer[map_features],
            map_layer_geometry='linestring',
            radius=radius,
        )
        lane_colors: npt.NDArray[np.uint8] = np.ones(len(lane_ids)).astype(np.uint8)
        baseline_paths_raster = _draw_linestring_image(
            image=baseline_paths_raster,
            object_coords=baseline_paths_coords,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=baseline_path_thickness,
            lane_colors=lane_colors,
        )

    # Flip the agents_raster along the horizontal axis.
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster


def get_baseline_paths_agents_raster(
    focus_agent: AgentState,
    map_api: AbstractMap,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    resolution: float,
    traffic_light_connectors: Dict[TrafficLightStatusType, List[str]],
    baseline_path_thickness: int = 1,
) -> npt.NDArray[np.float32]:
    """
    Construct the baseline paths layer by converting vector map to raster map.
    This function is for agents raster model, it has 3 channels for baseline path.
    :param focus_agent: agent state representing ego.
    :param map_api: map api
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param traffic_light_connectors: a dict mapping tl status type to a list of lane ids in this status.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :return baseline_paths_raster: the constructed baseline paths layer.
    """
    # Assume the raster has a square shape.
    if (x_range[1] - x_range[0]) != (y_range[1] - y_range[0]):
        raise ValueError(
            f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'
        )

    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster: npt.NDArray[np.float32] = np.zeros((*raster_shape, 3), dtype=np.float32)

    for map_features in ['LANE', 'LANE_CONNECTOR']:
        baseline_paths_coords, lane_ids = _get_layer_coords(
            agent=focus_agent,
            map_api=map_api,
            map_layer_name=SemanticMapLayer[map_features],
            map_layer_geometry='linestring',
            radius=radius,
        )

        # Get a list indicating the color of each lane
        lane_ids = np.asarray(lane_ids)  # type: ignore
        lane_colors: npt.NDArray[np.uint8] = np.full(
            (len(lane_ids), 3), BASELINE_TL_COLOR[TrafficLightStatusType.UNKNOWN], dtype=np.uint8
        )

        # If we have valid traffic light informaiton for intersection connectors
        if len(traffic_light_connectors) > 0:
            for tl_status in TrafficLightStatusType:
                if tl_status != TrafficLightStatusType.UNKNOWN and len(traffic_light_connectors[tl_status]) > 0:
                    lanes_in_tl_status = np.isin(lane_ids, traffic_light_connectors[tl_status])
                    lane_colors[lanes_in_tl_status] = BASELINE_TL_COLOR[tl_status]

        baseline_paths_raster = _draw_linestring_image(
            image=baseline_paths_raster,
            object_coords=baseline_paths_coords,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=baseline_path_thickness,
            lane_colors=lane_colors,
        )

    # Flip the agents_raster along the horizontal axis.
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster
