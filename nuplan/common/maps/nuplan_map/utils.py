from typing import Dict, List, Tuple, Union, cast

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely.geometry as geom

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map_objects import Lane, LaneConnector, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import RasterLayer, SemanticMapLayer, VectorLayer
from nuplan.database.maps_db.layer import MapLayer


def raster_layer_from_map_layer(map_layer: MapLayer) -> RasterLayer:
    """
    Convert MapDB's MapLayer to the generic RasterLayer.
    :param map_layer: input MapLayer object.
    :return: output RasterLayer object.
    """
    return RasterLayer(map_layer.data, map_layer.precision, map_layer.transform_matrix)


def split_blp_lane_segments(nodes: List[StateSE2], lane_seg_num: int) -> List[List[List[float]]]:
    """
    Split baseline path points into series of lane segment coordinate vectors.
    :param nodes: Baseline path nodes to be cut to lane_segments.
    :param lane_seg_num: Number of lane segments to split from baseline path.
    :return obj_coords: Data recording lane segment coordinates in format of [N, 2, 2].
    """
    obj_coords: List[List[List[float]]] = []

    for idx in range(lane_seg_num):
        # create segment vector from subsequent baseline path points
        curr_pt = [nodes[idx].x, nodes[idx].y]
        next_pt = [nodes[idx + 1].x, nodes[idx + 1].y]
        obj_coords.append([curr_pt, next_pt])

    return obj_coords


def connect_blp_lane_segments(start_lane_seg_idx: int, lane_seg_num: int) -> List[Tuple[int, int]]:
    """
    Add connection info for neighboring segments in baseline path.
    :param start_lane_seg_idx: Index for first lane segment in baseline path.
    :param lane_seg_num: Number of lane segments.
    :return obj_conns: Data recording lane-segment connection relations [from_lane_seg_idx, to_lane_seg_idx].
    """
    obj_conns: List[Tuple[int, int]] = []

    for lane_seg_idx in range(start_lane_seg_idx + 1, start_lane_seg_idx + lane_seg_num):
        obj_conns.append((lane_seg_idx - 1, lane_seg_idx))

    return obj_conns


def group_blp_lane_segments(start_lane_seg_idx: int, lane_seg_num: int) -> List[List[int]]:
    """
    Collect lane segment indices across lane/lane connector baseline path.
    :param start_lane_seg_idx: Index for first lane segment in baseline path.
    :param lane_seg_num: Number of lane segments.
    :return obj_groupings: Data recording lane-segment indices associated with given lane/lane connector.
    """
    obj_grouping: List[int] = []

    for lane_seg_idx in range(start_lane_seg_idx, start_lane_seg_idx + lane_seg_num):
        obj_grouping.append(lane_seg_idx)

    return [obj_grouping]


def trim_lane_nodes(point: Point2D, radius: float, lane_nodes: List[StateSE2]) -> List[StateSE2]:
    """
    Trim the discretized baseline path nodes to be within the radius. To ensure the continuity of
    the lane coords, only the end points of the lane/lane connectors are trimmed. For example, given
    the points in lane as [p_1, ..., p_n], only points at the end of the lane [p_1,...p_f], or
    [p_e, ... p_n] will be trimmed if they are further than the radius. The points between p_f and
    p_e will be kept regardless their distance to the ego.
    :param point: [m] x, y coordinates in global frame.
    :param radius [m] floating number about vector map query range.
    :param lane_nodes: The list of lane nodes to be filtered.
    :return obj_groupings: Data recording lane-segment indices associated with given lane/lane connector.
    """
    radius_squared = radius**2
    # Trim from the front end of lane/lane connector
    for index, node in enumerate(lane_nodes):
        if (node.x - point.x) ** 2 + (node.y - point.y) ** 2 <= radius_squared:
            start_index = index
            break
    else:
        return []
    # Trim from the other end of lane/lane connector
    for index, node in enumerate(lane_nodes[::-1]):
        if (node.x - point.x) ** 2 + (node.y - point.y) ** 2 <= radius_squared:
            end_index = len(lane_nodes) - index
            break

    return lane_nodes[start_index:end_index]


def build_lane_segments_from_blps_with_trim(
    point: Point2D, radius: float, map_obj: MapObject, start_lane_seg_idx: int
) -> Union[
    None, Tuple[List[List[List[float]]], List[Tuple[int, int]], List[List[int]], List[str], List[str], Tuple[int, int]]
]:
    """
    Process baseline paths of associated lanes/lane connectors to series of lane-segments along with connection info.
    :param point: [m] x, y coordinates in global frame.
    :param radius [m] floating number about vector map query range.
    :param map_obj: Lane or LaneConnector for building lane segments from associated baseline path.
    :param start_lane_seg_idx: Starting index for lane segments.
    :return
        obj_coords: Data recording lane-segment coordinates in format of [N, 2, 2].
        obj_conns: Data recording lane-segment connection relations in format of [M, 2].
        obj_groupings: Data recording lane-segment indices associated with each lane in format
            [num_lanes, num_segments_in_lane].
        obj_lane_ids: Data recording map object ids of lane/lane connector containing lane-segment.
        obj_roadblock_ids: Data recording map object ids of roadblock/roadblock connector containing lane-segment.
        obj_cross_blp_conn: Data storing indices of first and last lane segments of a given map object's baseline path
            as [blp_start_lane_seg_idx, blp_end_lane_seg_idx].
    """
    map_obj_id = map_obj.id
    roadblock_id = map_obj.get_roadblock_id()
    nodes = map_obj.baseline_path.discrete_path
    nodes = trim_lane_nodes(point, radius, nodes)

    # return None if at most one node are within the radius
    if len(nodes) <= 2:
        return None

    lane_seg_num = len(nodes) - 1
    end_lane_seg_idx = start_lane_seg_idx + lane_seg_num - 1

    obj_coords = split_blp_lane_segments(nodes, lane_seg_num)
    obj_conns = connect_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    obj_groupings = group_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    # record which map object each segment came from
    obj_lane_ids = [map_obj_id for _ in range(lane_seg_num)]
    # record which roadblock (lane group) each segment came from
    obj_roadblock_ids = [roadblock_id for _ in range(lane_seg_num)]
    # record first and last segment in baseline path for connecting lane and lane connectors later
    obj_cross_blp_conn = (start_lane_seg_idx, end_lane_seg_idx)

    return obj_coords, obj_conns, obj_groupings, obj_lane_ids, obj_roadblock_ids, obj_cross_blp_conn


def build_lane_segments_from_blps(
    map_obj: MapObject, start_lane_seg_idx: int
) -> Tuple[List[List[List[float]]], List[Tuple[int, int]], List[List[int]], List[str], List[str], Tuple[int, int]]:
    """
    Process baseline paths of associated lanes/lane connectors to series of lane-segments along with connection info.
    :param map_obj: Lane or LaneConnector for building lane segments from associated baseline path.
    :param start_lane_seg_idx: Starting index for lane segments.
    :return
        obj_coords: Data recording lane-segment coordinates in format of [N, 2, 2].
        obj_conns: Data recording lane-segment connection relations in format of [M, 2].
        obj_groupings: Data recording lane-segment indices associated with each lane in format
            [num_lanes, num_segments_in_lane].
        obj_lane_ids: Data recording map object ids of lane/lane connector containing lane-segment.
        obj_roadblock_ids: Data recording map object ids of roadblock/roadblock connector containing lane-segment.
        obj_cross_blp_conn: Data storing indices of first and last lane segments of a given map object's baseline path
            as [blp_start_lane_seg_idx, blp_end_lane_seg_idx].
    """
    map_obj_id = map_obj.id
    roadblock_id = map_obj.get_roadblock_id()
    nodes = map_obj.baseline_path.discrete_path
    lane_seg_num = len(nodes) - 1
    end_lane_seg_idx = start_lane_seg_idx + lane_seg_num - 1

    obj_coords = split_blp_lane_segments(nodes, lane_seg_num)
    obj_conns = connect_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    obj_groupings = group_blp_lane_segments(start_lane_seg_idx, lane_seg_num)
    # record which map object each segment came from
    obj_lane_ids = [map_obj_id for _ in range(lane_seg_num)]
    # record which roadblock (lane group) each segment came from
    obj_roadblock_ids = [roadblock_id for _ in range(lane_seg_num)]
    # record first and last segment in baseline path for connecting lane and lane connectors later
    obj_cross_blp_conn = (start_lane_seg_idx, end_lane_seg_idx)

    return obj_coords, obj_conns, obj_groupings, obj_lane_ids, obj_roadblock_ids, obj_cross_blp_conn


def connect_trimmed_lane_conn_predecessor(
    lane_coords: Tuple[List[List[List[float]]]],
    lane_conn: LaneConnector,
    cross_blp_conns: Dict[str, Tuple[int, int]],
    distance_threshold: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Given a specific lane connector, find its predecessor lane and return new connection info. To
                       handle the case where the end points of lane connector or/and the predecissor
                       lane being trimmed, a distance check is performed to make sure the end points
                       of the predecissor lane is close enough to be connected.
    :param: lane_coords: the lane segment cooridnates
    :param lane_conn: a specific lane connector.
    :param cross_blp_conns: Dict recording the map object id as key(str) and corresponding [first segment index,
        last segment index] pair as value (Tuple[int, int]).
    :param distance_threshold: the distance to determine if the end points are close enough to be
        connected in the lane graph.
    :return lane_seg_pred_conns: container recording the connection [from_lane_seg_idx, to_lane_seg_idx] between
        last predecessor segment and first segment of given lane connector.
    """
    lane_seg_pred_conns: List[Tuple[int, int]] = []
    lane_conn_start_seg_idx, lane_conn_end_seg_idx = cross_blp_conns[lane_conn.id]
    incoming_lanes = [incoming_edge for incoming_edge in lane_conn.incoming_edges if isinstance(incoming_edge, Lane)]

    for incoming_lane in incoming_lanes:
        lane_id = incoming_lane.id

        if lane_id in cross_blp_conns.keys():
            # record connection between last segment of incoming lane and first segment of given lane connector
            predecessor_start_idx, predecessor_end_idx = cross_blp_conns[lane_id]
            if (
                np.linalg.norm(
                    np.array(lane_coords[predecessor_end_idx][1]) - np.array(lane_coords[lane_conn_start_seg_idx][0])
                )
                < distance_threshold
            ):
                lane_seg_pred_conns.append((predecessor_end_idx, lane_conn_start_seg_idx))

    return lane_seg_pred_conns


def connect_trimmed_lane_conn_successor(
    lane_coords: Tuple[List[List[List[float]]]],
    lane_conn: LaneConnector,
    cross_blp_conns: Dict[str, Tuple[int, int]],
    distance_threshold: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Given a specific lane connector, find its successor lane and return new connection info. To
                       handle the case where the end points of lane connector or/and the predecissor
                       lane being trimmed, a distance check is performed to make sure the end points
                       of the predecissor lane is close enough to be connected.
    :param: lane_coords: the lane segment cooridnates
    :param lane_conn: a specific lane connector.
    :param cross_blp_conns: Dict recording the map object id as key(str) and corresponding [first segment index,
        last segment index] pair as value (Tuple[int, int]).
    :param distance_threshold: the distance to determine if the end points are close enough to be
        connected in the lane graph.
    :return lane_seg_suc_conns: container recording the connection [from_lane_seg_idx, to_lane_seg_idx] between
        last segment of given lane connector and first successor lane segment.
    """
    lane_seg_suc_conns: List[Tuple[int, int]] = []
    lane_conn_start_seg_idx, lane_conn_end_seg_idx = cross_blp_conns[lane_conn.id]
    outgoing_lanes = [outgoing_edge for outgoing_edge in lane_conn.outgoing_edges if isinstance(outgoing_edge, Lane)]

    for outgoing_lane in outgoing_lanes:
        lane_id = outgoing_lane.id

        if lane_id in cross_blp_conns.keys():
            # record connection between last segment of given lane connector and first segment of outgoing lane
            successor_start_idx, successor_end_seg_idx = cross_blp_conns[lane_id]
            if (
                np.linalg.norm(
                    np.array(lane_coords[lane_conn_end_seg_idx][1]) - np.array(lane_coords[successor_start_idx][0])
                )
                < distance_threshold
            ):
                lane_seg_suc_conns.append((lane_conn_end_seg_idx, successor_start_idx))

    return lane_seg_suc_conns


def connect_lane_conn_predecessor(
    lane_conn: LaneConnector, cross_blp_conns: Dict[str, Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Given a specific lane connector, find its predecessor lane and return new connection info.
    :param lane_conn: a specific lane connector.
    :param cross_blp_conns: Dict recording the map object id as key(str) and corresponding [first segment index,
        last segment index] pair as value (Tuple[int, int]).
    :return lane_seg_pred_conns: container recording the connection [from_lane_seg_idx, to_lane_seg_idx] between
        last predecessor segment and first segment of given lane connector.
    """
    lane_seg_pred_conns: List[Tuple[int, int]] = []
    lane_conn_start_seg_idx, lane_conn_end_seg_idx = cross_blp_conns[lane_conn.id]
    incoming_lanes = [incoming_edge for incoming_edge in lane_conn.incoming_edges if isinstance(incoming_edge, Lane)]

    for incoming_lane in incoming_lanes:
        lane_id = incoming_lane.id

        if lane_id in cross_blp_conns.keys():
            # record connection between last segment of incoming lane and first segment of given lane connector
            predecessor_start_idx, predecessor_end_idx = cross_blp_conns[lane_id]
            lane_seg_pred_conns.append((predecessor_end_idx, lane_conn_start_seg_idx))

    return lane_seg_pred_conns


def connect_lane_conn_successor(
    lane_conn: LaneConnector, cross_blp_conns: Dict[str, Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Given a specific lane connector, find its successor lane and return new connection info.
    :param lane_conn: a specific lane connector.
    :param cross_blp_conns: Dict recording the map object id as key(str) and corresponding [first segment index,
        last segment index] pair as value (Tuple[int, int]).
    :return lane_seg_suc_conns: container recording the connection [from_lane_seg_idx, to_lane_seg_idx] between
        last segment of given lane connector and first successor lane segment.
    """
    lane_seg_suc_conns: List[Tuple[int, int]] = []
    lane_conn_start_seg_idx, lane_conn_end_seg_idx = cross_blp_conns[lane_conn.id]
    outgoing_lanes = [outgoing_edge for outgoing_edge in lane_conn.outgoing_edges if isinstance(outgoing_edge, Lane)]

    for outgoing_lane in outgoing_lanes:
        lane_id = outgoing_lane.id

        if lane_id in cross_blp_conns.keys():
            # record connection between last segment of given lane connector and first segment of outgoing lane
            successor_start_idx, successor_end_seg_idx = cross_blp_conns[lane_id]
            lane_seg_suc_conns.append((lane_conn_end_seg_idx, successor_start_idx))

    return lane_seg_suc_conns


def extract_polygon_from_map_object(map_object: MapObject) -> List[Point2D]:
    """
    Extract polygon from map object.
    :param map_object: input MapObject.
    :return: polygon as list of Point2D.
    """
    x_coords, y_coords = map_object.polygon.exterior.coords.xy
    return [Point2D(x, y) for x, y in zip(x_coords, y_coords)]


def extract_roadblock_objects(map_api: AbstractMap, point: Point2D) -> List[RoadBlockGraphEdgeMapObject]:
    """
    Extract roadblock or roadblock connectors from map containing point if they exist.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :return List of roadblocks/roadblock connectors containing point if they exist.
    """
    roadblock = map_api.get_one_map_object(point, SemanticMapLayer.ROADBLOCK)
    if roadblock:

        return [roadblock]
    else:
        roadblock_conns = map_api.get_all_map_objects(point, SemanticMapLayer.ROADBLOCK_CONNECTOR)

        return cast(List[RoadBlockGraphEdgeMapObject], roadblock_conns)


def get_roadblock_ids_from_trajectory(map_api: AbstractMap, ego_states: List[EgoState]) -> List[str]:
    """
    Extract ids of roadblocks and roadblock connectors containing points in specified trajectory.
    :param map_api: map to perform extraction on.
    :param ego_states: sequence of agent states representing trajectory.
    :return roadblock_ids: List of ids of roadblocks/roadblock connectors containing trajectory points.
    """
    roadblock_ids: List[str] = []
    roadblock_candidates: List[RoadBlockGraphEdgeMapObject] = []
    last_roadblock = None
    points = [ego_state.rear_axle.point for ego_state in ego_states]

    for point in points:

        # skip repeated roadblocks
        if last_roadblock and last_roadblock.contains_point(point):
            continue

        # if no candidates under consideration, use graph search from last element in route to find next candidates
        if last_roadblock and not roadblock_candidates:
            roadblock_candidates = last_roadblock.outgoing_edges

        # refine candidates if existing to those containing current point
        roadblock_candidates = [roadblock for roadblock in roadblock_candidates if roadblock.contains_point(point)]

        # if singular candidate remains, add it to route
        if len(roadblock_candidates) == 1:
            last_roadblock = roadblock_candidates.pop()
            roadblock_ids.append(last_roadblock.id)
        # if no valid candidates, expand search to whole map
        elif not roadblock_candidates:
            roadblock_objects = extract_roadblock_objects(map_api, point)

            if len(roadblock_objects) == 1:
                last_roadblock = roadblock_objects.pop()
                roadblock_ids.append(last_roadblock.id)
            else:
                roadblock_candidates = roadblock_objects

    return roadblock_ids


def is_in_type(x: float, y: float, vector_layer: VectorLayer) -> bool:
    """
    Checks if position [x, y] is in any entry of type.
    :param x: [m] floating point x-coordinate in global frame.
    :param y: [m] floating point y-coordinate in global frame.
    :param vector_layer: vector layer to be searched through.
    :return True iff position [x, y] is in any entry of type, False if it is not.
    """
    assert vector_layer is not None, "type can not be None!"

    in_polygon = vector_layer.contains(geom.Point(x, y))

    return any(in_polygon.values)


def get_all_rows_with_value(
    elements: gpd.geodataframe.GeoDataFrame, column_label: str, desired_value: str
) -> gpd.geodataframe.GeoDataFrame:
    """
    Extract all matching elements. Note, if no matching desired_key is found and empty list is returned.
    :param elements: data frame from MapsDb.
    :param column_label: key to extract from a column.
    :param desired_value: key which is compared with the values of column_label entry.
    :return: a subset of the original GeoDataFrame containing the matching key.
    """
    return elements.iloc[np.where(elements[column_label].to_numpy().astype(int) == int(desired_value))]


def get_row_with_value(elements: gpd.geodataframe.GeoDataFrame, column_label: str, desired_value: str) -> pd.Series:
    """
    Extract a matching element.
    :param elements: data frame from MapsDb.
    :param column_label: key to extract from a column.
    :param desired_value: key which is compared with the values of column_label entry.
    :return row from GeoDataFrame.
    """
    if column_label == "fid":
        return elements.loc[desired_value]

    matching_rows = get_all_rows_with_value(elements, column_label, desired_value)
    assert len(matching_rows) > 0, f"Could not find the desired key = {desired_value}"
    assert len(matching_rows) == 1, (
        f"{len(matching_rows)} matching keys found. Expected to only find one." "Try using get_all_rows_with_value"
    )

    return matching_rows.iloc[0]


def compute_linestring_heading(linestring: geom.linestring.LineString) -> List[float]:
    """
    Compute the heading of each coordinate to its successor coordinate. The last coordinate will have the same heading
        as the second last coordinate.
    :param linestring: linestring as a shapely LineString.
    :return: a list of headings associated to each starting coordinate.
    """
    coords: npt.NDArray[np.float64] = np.asarray(linestring.coords)
    vectors = np.diff(coords, axis=0)
    angles = np.arctan2(vectors.T[1], vectors.T[0])
    angles = np.append(angles, angles[-1])  # pad end with duplicate heading

    assert len(angles) == len(coords), "Calculated heading must have the same length as input coordinates"

    return list(angles)


def compute_curvature(point1: geom.Point, point2: geom.Point, point3: geom.Point) -> float:
    """
    Estimate signed curvature along the three points.
    :param point1: First point of a circle.
    :param point2: Second point of a circle.
    :param point3: Third point of a circle.
    :return signed curvature of the three points.
    """
    # points_utm is a 3-by-2 array, containing the easting and northing coordinates of 3 points
    # Compute distance to each point
    a = point1.distance(point2)
    b = point2.distance(point3)
    c = point3.distance(point1)

    # Compute inverse radius of circle using surface of triangle (for which Heron's formula is used)
    surface_2 = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))

    if surface_2 < 1e-6:
        # In this case the points are almost aligned in a lane
        return 0.0

    assert surface_2 >= 0
    k = np.sqrt(surface_2) / 4  # Heron's formula for triangle's surface
    den = a * b * c  # Denumerator; make sure there is no division by zero.
    curvature = 4 * k / den if not np.isclose(den, 0.0) else 0.0

    # The curvature is unsigned, in order to extract sign, the third point is checked wrt to point1-point2 line
    position = np.sign((point2.x - point1.x) * (point3.y - point1.y) - (point2.y - point1.y) * (point3.x - point1.x))

    return float(position * curvature)


def get_distance_between_map_object_and_point(point: Point2D, map_object: MapObject) -> float:
    """
    Get distance between point and nearest surface of specified map object.
    :param point: Point to calculate distance between.
    :param map_object: MapObject (containing underlying polygon) to check distance between.
    :return: Computed distance.
    """
    return float(geom.Point(point.x, point.y).distance(map_object.polygon))


def extract_discrete_polyline(polyline: geom.LineString) -> List[StateSE2]:
    """
    Returns a discretized polyline composed of StateSE2 as nodes.
    :param polyline: the polyline of interest.
    :returns: linestring as a list of waypoints represented by StateSE2.
    """
    assert polyline.length > 0.0, "The length of the polyline has to be greater than 0!"

    headings = compute_linestring_heading(polyline)
    x_coords, y_coords = polyline.coords.xy

    return [StateSE2(x, y, heading) for x, y, heading in zip(x_coords, y_coords, headings)]


def estimate_curvature_along_path(
    path: geom.LineString, arc_length: float, distance_for_curvature_estimation: float
) -> float:
    """
    Estimate curvature along a path at arc_length from origin.
    :param path: LineString creating a continuous path.
    :param arc_length: [m] distance from origin of the path.
    :param distance_for_curvature_estimation: [m] the distance used to construct 3 points.
    :return estimated curvature at point arc_length.
    """
    assert 0 <= arc_length <= path.length

    # Extract 3 points from a path
    if path.length < 2.0 * distance_for_curvature_estimation:
        # In this case the arch_length is too short
        first_arch_length = 0.0
        second_arc_length = path.length / 2.0
        third_arc_length = path.length
    elif arc_length - distance_for_curvature_estimation < 0.0:
        # In this case the arch_length is too close to origin
        first_arch_length = 0.0
        second_arc_length = distance_for_curvature_estimation
        third_arc_length = 2.0 * distance_for_curvature_estimation
    elif arc_length + distance_for_curvature_estimation > path.length:
        # In this case the arch_length is too close to end of the path
        first_arch_length = path.length - 2.0 * distance_for_curvature_estimation
        second_arc_length = path.length - distance_for_curvature_estimation
        third_arc_length = path.length
    else:  # In this case the arc_length lands along the path
        first_arch_length = arc_length - distance_for_curvature_estimation
        second_arc_length = arc_length
        third_arc_length = arc_length + distance_for_curvature_estimation

    first_arch_position = path.interpolate(first_arch_length)
    second_arch_position = path.interpolate(second_arc_length)
    third_arch_position = path.interpolate(third_arc_length)

    return compute_curvature(first_arch_position, second_arch_position, third_arch_position)
