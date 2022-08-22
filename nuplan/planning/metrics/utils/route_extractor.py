from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

import numpy as np
import numpy.typing as npt
from shapely.geometry import Point

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    GraphEdgeMapObject,
    Lane,
    LaneConnector,
    LaneGraphEdgeMapObject,
    PolylineMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

logger = logging.getLogger(__name__)


@dataclass
class RouteBaselineRoadBlockPair:
    """
    Class for storing the corresponding road_block and baseline of a lane/lane_connector
    :param road_block: A lane group or a lane group connector
    :param base_line: A base line path.
    """

    road_block: LaneGraphEdgeMapObject
    base_line: PolylineMapObject
    next: Optional[RouteBaselineRoadBlockPair] = None


@dataclass
class RouteRoadBlockLinkedList:
    """
    A linked list of RouteBaselineRoadBlockPairs
    :param head: Head of the linked list, defaults to None.
    """

    head: Optional[RouteBaselineRoadBlockPair] = None


def get_current_route_objects(map_api: AbstractMap, pose: Point2D) -> List[GraphEdgeMapObject]:
    """
    Gets the list including the lane or lane_connectors the pose corresponds to if there exists one, and empty list o.w
    :param map_api: map
    :param pose: xy coordinates
    :return the corresponding route object.
    """
    curr_lane = map_api.get_one_map_object(pose, SemanticMapLayer.LANE)
    if curr_lane is None:
        # Get the list of lane connectors if exists, otherwise it returns and empty list
        curr_lane_connectors = map_api.get_all_map_objects(pose, SemanticMapLayer.LANE_CONNECTOR)
        route_objects_with_pose = curr_lane_connectors
    else:
        route_objects_with_pose = [curr_lane]

    return route_objects_with_pose  # type: ignore


def get_route_obj_with_candidates(
    pose: Point2D, candidate_route_objs: List[GraphEdgeMapObject]
) -> List[GraphEdgeMapObject]:
    """
    This function uses a candidate set of lane/lane-connectors and return the lane/lane-connector that correponds to the pose
    by checking if pose belongs to one of the route objs in candidate_route_objs or their outgoing_edges
    :param pose: ego_pose
    :param candidate_route_objs: a list of route objects
    :return: a list of route objects corresponding to the pose
    """
    if not len(candidate_route_objs):
        raise ValueError('candidate_route_objs list is empty, no candidates to start with')

    # for each pose first check if pose belongs to candidate route objs
    route_objects_with_pose = [
        one_route_obj for one_route_obj in candidate_route_objs if one_route_obj.contains_point(pose)
    ]

    # if it does not, and candidate set has only one element check wether it's in an outgoing_edge of the previous lane/lane_connector.
    # It is expected that ego is eventually assigned to a single lane-connector when it is entering an outgoing_edge, and hence the logic:
    if not route_objects_with_pose and len(candidate_route_objs) == 1:
        route_objects_with_pose = [
            next_route_obj
            for next_route_obj in candidate_route_objs[0].outgoing_edges
            if next_route_obj.contains_point(pose)
        ]
    return route_objects_with_pose


def remove_extra_lane_connectors(route_objs: List[List[GraphEdgeMapObject]]) -> List[List[GraphEdgeMapObject]]:
    """
    # This function iterate through route object and replace field with multiple lane_connectors
    # with the one lane_connector ego ends up in.
    :param route_objs: a list of route objects.
    """
    # start from last object in the route list
    last_to_first_route_list = route_objs[::-1]
    enum = enumerate(last_to_first_route_list)
    for ind, curr_last_obj in enum:
        # skip if ind = 0 or if there's a single object in current objects list
        if ind == 0 or len(curr_last_obj) <= 1:
            continue
        # O.w cull down the curr_last_obj using the next obj (prev obj in the reversed list) if possible
        if len(curr_last_obj) > len(last_to_first_route_list[ind - 1]):
            curr_route_obj_ids = [obj.id for obj in curr_last_obj]
            if all([(obj.id in curr_route_obj_ids) for obj in last_to_first_route_list[ind - 1]]):
                last_to_first_route_list[ind] = last_to_first_route_list[ind - 1]
        # Skip the rest if there's no more than one object left
        if len(curr_last_obj) <= 1:
            continue
        # Otherwise try to see if you can cull down lane_connectors using the lane ego ends up in and its incoming_edges
        if last_to_first_route_list[ind - 1] and isinstance(last_to_first_route_list[ind - 1][0], Lane):
            next_lane_incoming_edge_ids = [obj.id for obj in last_to_first_route_list[ind - 1][0].incoming_edges]
            objs_to_keep = [obj for obj in curr_last_obj if obj.id in next_lane_incoming_edge_ids]
            if objs_to_keep:
                last_to_first_route_list[ind] = objs_to_keep

    return last_to_first_route_list[::-1]


def get_route(map_api: AbstractMap, poses: List[Point2D]) -> List[List[GraphEdgeMapObject]]:
    """
    Returns and sets the sequence of lane and lane connectors corresponding to the trajectory
    :param map_api: map
    :param poses: a list of xy coordinates
    :return list of route objects.
    """
    if not len(poses):
        raise ValueError('invalid poses passed to get_route()')

    route_objs: List[List[GraphEdgeMapObject]] = []

    # Find the lane/lane_connector ego belongs to initially
    curr_route_obj: List[GraphEdgeMapObject] = []

    for ind, pose in enumerate(poses):
        if curr_route_obj:
            # next, for each pose first check if pose belongs to previously found lane/lane_connectors,
            # if it does not, check wether it's in an outgoing_egde of the previous lane/lane_connector
            curr_route_obj = get_route_obj_with_candidates(pose, curr_route_obj)

        # If route obj is not found using the previous step re-search the map
        if not curr_route_obj:
            curr_route_obj = get_current_route_objects(map_api, pose)
            # Ideally, two successive lane_connectors in the list shouldn't be distinct. However in some cases trajectory can slightly goes outside the
            # associated lane_connector and lies inside an irrelevant lane_connector. Filter these cases if pose is still close to the previous lane_connector:
            if (
                ind > 1
                and route_objs[-1]
                and isinstance(route_objs[-1][0], LaneConnector)
                and (
                    (curr_route_obj and isinstance(curr_route_obj[0], LaneConnector))
                    or (not curr_route_obj and map_api.is_in_layer(pose, SemanticMapLayer.INTERSECTION))
                )
            ):
                previous_proximal_route_obj = [obj for obj in route_objs[-1] if obj.polygon.distance(Point(*pose)) < 5]

                if previous_proximal_route_obj:
                    curr_route_obj = previous_proximal_route_obj
        route_objs.append(curr_route_obj)

    # iterate through route object and replace field with multiple lane_connectors with the one lane_connector ego ends up in.
    improved_route_obj = remove_extra_lane_connectors(route_objs)
    return improved_route_obj


def get_route_simplified(route_list: List[List[LaneGraphEdgeMapObject]]) -> List[List[LaneGraphEdgeMapObject]]:
    """
    This function simplifies the route by removing repeated consequtive route objects
    :param route_list: A list of route objects representing ego's corresponding route objects at each instance
    :return A simplified list of route objects that shows the order of route objects ego has been in.
    """
    # This function captures lane/lane_connector changes and does not necessarily return unique route objects
    try:
        # Find the first pose that belongs to a lane/lane_connector
        ind = next(iteration for iteration, iter_route_list in enumerate(route_list) if iter_route_list)
    except StopIteration:
        logger.warning('All route_list elements are empty, returning an empty list from get_route_simplified()')
        return []

    route_simplified = [route_list[ind]]
    for route_object in route_list[ind + 1 :]:
        repeated_entries = [
            obj_id
            for obj_id in [prev_obj.id for prev_obj in route_simplified[-1]]
            if obj_id in [one_route_obj.id for one_route_obj in route_object]
        ]
        if route_object and not repeated_entries:
            route_simplified.append(route_object)
    return route_simplified


def get_route_baseline_roadblock_linkedlist(
    map_api: AbstractMap, expert_route: List[List[LaneGraphEdgeMapObject]]
) -> RouteRoadBlockLinkedList:
    """
    This function generates a linked list of baseline & unique road-block pairs
    (RouteBaselineRoadBlockPair) from a simplified route
    :param map_api: Corresponding map
    :param expert_route: A route list
    :return A linked list of RouteBaselineRoadBlockPair.
    """
    # Simplification: Take the first route obj even if there are multiple ones
    # Assumption: Expert is always in a lane or a lane_connector

    route_baseline_roadblock_list = RouteRoadBlockLinkedList()
    prev_roadblock_id = None

    for route_object in expert_route:
        if route_object:
            # simply get the first route obj
            roadblock_id = route_object[0].get_roadblock_id()
            if roadblock_id != prev_roadblock_id:
                prev_roadblock_id = roadblock_id
                if isinstance(route_object[0], Lane):
                    road_block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
                else:
                    road_block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)

                ref_baseline_path = route_object[0].baseline_path

                if route_baseline_roadblock_list.head is None:
                    prev_route_baseline_roadblock = RouteBaselineRoadBlockPair(
                        base_line=ref_baseline_path, road_block=road_block
                    )
                    route_baseline_roadblock_list.head = prev_route_baseline_roadblock
                else:
                    prev_route_baseline_roadblock.next = RouteBaselineRoadBlockPair(
                        base_line=ref_baseline_path, road_block=road_block
                    )
                    prev_route_baseline_roadblock = prev_route_baseline_roadblock.next

    return route_baseline_roadblock_list


def get_distance_of_closest_baseline_point_to_its_start(base_line: PolylineMapObject, pose: Point2D) -> float:
    """Computes distance of "closest point on the baseline to pose" to the beginning of the baseline
    :param base_line: A baseline path
    :param pose: An ego pose
    :return: distance to start.
    """
    return float(base_line.linestring.project(Point(*pose)))


@dataclass
class CornersGraphEdgeMapObject:
    """class containing list of lane/lane connectors of each corner of ego's orientedbox."""

    front_left_map_objs: List[GraphEdgeMapObject]
    rear_left_map_objs: List[GraphEdgeMapObject]
    rear_right_map_objs: List[GraphEdgeMapObject]
    front_right_map_objs: List[GraphEdgeMapObject]

    def __iter__(self) -> Iterable[List[GraphEdgeMapObject]]:
        """Returns an iterable tuple of the class attributes
        :return: iterator of class attribute tuples
        """
        return iter(
            (self.front_left_map_objs, self.rear_left_map_objs, self.rear_right_map_objs, self.front_right_map_objs)
        )


def extract_corners_route(
    map_api: AbstractMap, ego_footprint_list: List[OrientedBox]
) -> List[CornersGraphEdgeMapObject]:
    """
    Extracts lists of lane/lane connectors of corners of ego from history
    :param map_api: AbstractMap
    :param ego_corners_list: List of OrientedBoxes
    :return List of CornersGraphEdgeMapObject class containing list of lane/lane connectors of each
    corner of ego in the history.
    """
    if not len(ego_footprint_list):
        logger.warning('Invalid poses passed to extract_corners_route()')
        return []

    corners_route: List[CornersGraphEdgeMapObject] = []
    curr_candid_route_obj: List[GraphEdgeMapObject] = []

    for ind, ego_footprint in enumerate(ego_footprint_list):
        # Initialize the obj to store the corners' lane/lane_conns in this iteration
        corners_route_objs = CornersGraphEdgeMapObject([], [], [], [])
        ego_corners = ego_footprint.all_corners()
        next_candid_route_obj: List[GraphEdgeMapObject] = []  # candid route set for the next iteration
        for ego_corner, corner_type in zip(ego_corners, corners_route_objs.__dict__.keys()):
            # Use the corners route candidate set to find the route for the corner
            route_object = []
            if curr_candid_route_obj:
                route_object = get_route_obj_with_candidates(ego_corner, curr_candid_route_obj)

            # If route obj wasn't found using the candidate set, re-search the map
            if not route_object:
                route_object = get_current_route_objects(map_api, ego_corner)
                # For the first pose, update the candidate set when route object is found for one corner
                if ind == 1:
                    curr_candid_route_obj += [
                        obj for obj in route_object if obj.id not in [candid.id for candid in curr_candid_route_obj]
                    ]
            # Add route objects that corners belong to to the candidate set for the next iteration
            next_candid_route_obj += [
                obj for obj in route_object if obj.id not in [candid.id for candid in next_candid_route_obj]
            ]

            # Set the corresponding attribute
            corners_route_objs.__setattr__(corner_type, route_object)

        corners_route.append(corners_route_objs)

        # Update the corners candidate set
        curr_candid_route_obj = next_candid_route_obj

    return corners_route


def get_outgoing_edges_obj_dict(corner_route_object: List[GraphEdgeMapObject]) -> dict[str, GraphEdgeMapObject]:
    """
    :param corner_route_object: List of lane/lane connectors
    :return dictionary of id and itscorresponding route object of outgoing edges of a given route object
    """
    return {obj_edge.id: obj_edge for obj in corner_route_object for obj_edge in obj.outgoing_edges}


def get_incoming_edges_obj_dict(corner_route_object: List[GraphEdgeMapObject]) -> dict[str, GraphEdgeMapObject]:
    """
    :param corner_route_object: List of lane/lane connectors
    :return dictionary of id and itscorresponding route object of incoming edges of a given route object
    """
    return {obj_edge.id: obj_edge for obj in corner_route_object for obj_edge in obj.incoming_edges}


def get_common_route_object(
    corners_route_obj_ids: List[Set[str]], obj_id_dict: dict[str, GraphEdgeMapObject]
) -> Set[GraphEdgeMapObject]:
    """
    Extracts common lane/lane connectors of corners
    :param corners_route_obj_ids: List of ids of route objects of corners of ego
    :param obj_id_dict: dictionary of ids and corresponding route objects
    :return set of common route objects, returns an empty set of no common object is found.
    """
    return {obj_id_dict[id] for id in set.intersection(*corners_route_obj_ids)}


def get_connecting_route_object(
    corners_route_obj_list: List[List[GraphEdgeMapObject]],
    corners_route_obj_ids: List[Set[str]],
    obj_id_dict: dict[str, GraphEdgeMapObject],
) -> Set[GraphEdgeMapObject]:
    """
    Extracts connecting (outgoing or incoming) lane/lane connectors of corners
    :param corners_route_obj_list: List of route objects of corners of ego
    :param corners_route_obj_ids: List of ids of route objects of corners of ego
    :param obj_id_dict: dictionary of ids and corresponding route objects
    :return set of connecting route objects, returns an empty set of no connecting object is found.
    """
    all_corners_connecting_obj_ids = set()

    front_left_route_obj, rear_left_route_obj, rear_right_route_obj, front_right_route_obj = corners_route_obj_list
    (
        front_left_route_obj_ids,
        rear_left_route_obj_ids,
        rear_right_route_obj_ids,
        front_right_route_obj_ids,
    ) = corners_route_obj_ids

    rear_right_route_obj_out_edge_dict = get_outgoing_edges_obj_dict(rear_right_route_obj)
    rear_left_route_obj_out_edge_dict = get_outgoing_edges_obj_dict(rear_left_route_obj)
    # Update dictionary of id: route object
    obj_id_dict = {**obj_id_dict, **rear_right_route_obj_out_edge_dict, **rear_left_route_obj_out_edge_dict}

    # Check if a rear corner is in the same or a connected outgoing lane/lane connectors of the other rear corner
    rear_right_obj_or_outgoing_edge = rear_right_route_obj_ids.union(set(rear_right_route_obj_out_edge_dict.keys()))
    rear_left_in_rear_right_obj_or_outgoing_edge = rear_left_route_obj_ids.intersection(rear_right_obj_or_outgoing_edge)

    rear_left_obj_or_outgoing_edge = rear_left_route_obj_ids.union(set(rear_left_route_obj_out_edge_dict.keys()))
    rear_right_in_rear_left_obj_or_outgoing_edge = rear_right_route_obj_ids.intersection(rear_left_obj_or_outgoing_edge)

    rear_corners_connecting_obj_ids = rear_left_in_rear_right_obj_or_outgoing_edge.union(
        rear_right_in_rear_left_obj_or_outgoing_edge
    )

    if len(rear_corners_connecting_obj_ids) > 0:
        # Check if the right/left front corners are in the same or a connected lane/lane connectors
        # of the opposite (left/right) rear corners
        front_left_route_obj_in_edge_dict = get_incoming_edges_obj_dict(front_left_route_obj)
        front_left_obj_or_incoming_edge = front_left_route_obj_ids.union(set(front_left_route_obj_in_edge_dict.keys()))
        front_left_rear_right_common_obj_ids = front_left_obj_or_incoming_edge.intersection(
            rear_right_obj_or_outgoing_edge
        )

        front_right_route_obj_in_edge_dict = get_incoming_edges_obj_dict(front_right_route_obj)
        front_right_obj_or_incoming_edge = front_right_route_obj_ids.union(
            set(front_right_route_obj_in_edge_dict.keys())
        )
        front_right_rear_left_common_obj_ids = front_right_obj_or_incoming_edge.intersection(
            rear_left_obj_or_outgoing_edge
        )

        all_corners_connecting_obj_ids = {
            obj_id_dict[id]
            for id in set.intersection(front_left_rear_right_common_obj_ids, front_right_rear_left_common_obj_ids)
        }

    return all_corners_connecting_obj_ids


def extract_common_or_connecting_route_objs(
    corners_route_obj: CornersGraphEdgeMapObject,
) -> Optional[Set[GraphEdgeMapObject]]:
    """
    Extracts common or connecting (outgoing or incoming) lane/lane connectors of corners
    :param corners_route_obj: Class containing list of lane/lane connectors of each corner
    :return common or connecting lane/lane connectors of corners if exists, else None.
    If all corners are in nondrivable area, returns an empty set.
    """
    corners_route_obj_list = [*corners_route_obj.__iter__()]

    not_in_lane_or_laneconn = [
        True if len(corner_route_obj) == 0 else False for corner_route_obj in corners_route_obj_list
    ]

    # Check if all corners are outside lane/lane_connectors
    if np.all(not_in_lane_or_laneconn):
        return set()

    # Check if any corner (not all of them) is outside lane/lane_connectors
    if np.any(not_in_lane_or_laneconn):
        return None

    # Keep a dictionary of ids of lanes/lane connectors to later retrieve them using their ids
    obj_id_dict = {obj.id: obj for corner_route_obj in corners_route_obj_list for obj in corner_route_obj}
    corners_route_obj_ids = [{obj.id for obj in corner_route_obj} for corner_route_obj in corners_route_obj_list]

    # Check if corners are in the same lane/lane connector
    all_corners_common_obj = get_common_route_object(corners_route_obj_ids, obj_id_dict)
    if len(all_corners_common_obj) > 0:
        # Return the set of common lane/lane connectors
        return all_corners_common_obj

    # Check if corners are in a connecting (outgoing or incoming) lane/lane connectors
    all_corners_connecting_obj = get_connecting_route_object(corners_route_obj_list, corners_route_obj_ids, obj_id_dict)
    if len(all_corners_connecting_obj) > 0:
        return all_corners_connecting_obj

    # If no common or conecting route object is found, return None
    return None


def get_timestamps_in_common_or_connected_route_objs(
    common_or_connected_route_objs: List[Optional[Set[GraphEdgeMapObject]]], ego_timestamps: npt.NDArray[np.int32]
) -> List[int]:
    """
    Extract timestamps when ego's corners are in common or connected lane/lane connectors.
    :param common_or_connected_route_objs: list of common or connected lane/lane connectors of corners if exist,
    empty list if all corners are in non_drivable area and None if corners are in different lane/lane connectors
    :param ego_timestamps: Array of times in time_us
    :return List of ego_timestamps where all corners of ego are in common or connected route objects
    """
    return [timestamp for route_obj, timestamp in zip(common_or_connected_route_objs, ego_timestamps) if route_obj]


def get_common_or_connected_route_objs_of_corners(
    corners_route: List[CornersGraphEdgeMapObject],
) -> List[Optional[Set[GraphEdgeMapObject]]]:
    """
    Returns a list of common or connected lane/lane connectors of corners.
    :param corners_route: List of class conatining list of lane/lane connectors of corners of ego
    :return list of common or connected lane/lane connectors of corners if exist, empty list if all corners are
    in non_drivable area and None if corners are in different lane/lane connectors.
    """
    history_common_or_connecting_route_objs: List[Optional[Set[GraphEdgeMapObject]]] = []

    prev_corners_route_obj = corners_route[0]

    corners_common_or_connecting_route_objs = extract_common_or_connecting_route_objs(prev_corners_route_obj)
    history_common_or_connecting_route_objs.append(corners_common_or_connecting_route_objs)

    for curr_corners_route_obj in corners_route[1:]:

        if curr_corners_route_obj != prev_corners_route_obj:

            corners_common_or_connecting_route_objs = extract_common_or_connecting_route_objs(curr_corners_route_obj)

        history_common_or_connecting_route_objs.append(corners_common_or_connecting_route_objs)

        prev_corners_route_obj = curr_corners_route_obj

    return history_common_or_connecting_route_objs
