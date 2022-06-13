from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import BaselinePath, GraphEdgeMapObject, Lane, LaneGraphEdgeMapObject
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
    base_line: BaselinePath
    next: Optional[RouteBaselineRoadBlockPair] = None


@dataclass
class RouteRoadBlockLinkedList:
    """
    A linked list of RouteBaselineRoadBlockPairs
    :param head: Head of the linked list, defaults to None.
    """

    head: Optional[RouteBaselineRoadBlockPair] = None


def get_curr_route_obj(map_api: AbstractMap, pose: Point2D) -> List[GraphEdgeMapObject]:
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
        curr_route_obj = curr_lane_connectors
    else:
        curr_route_obj = [curr_lane]

    return curr_route_obj  # type: ignore


def get_route(map_api: AbstractMap, poses: List[Point2D]) -> List[List[GraphEdgeMapObject]]:
    """
    Returns and sets the sequence of lane and lane connectors corresponding to the trajectory
    :param map_api: map
    :param poses: a list of xy coordinates
    :return list of route objects.
    """
    route_objs: List[List[GraphEdgeMapObject]] = []

    if not len(poses):
        logger.warning('Invalid poses passed to get_route()')
        return route_objs

    # Find the lane/lane_connector ego belongs to initially
    curr_route_obj = get_curr_route_obj(map_api, poses[0])
    route_objs.append(curr_route_obj)

    # After finding the initial lane/lane_connector, for each pose first check if pose belongs to previously found
    # lane/lane_connector, if it does not check wether it's in an outgoing_egde of the previous lane/lane_connector,
    # otherwise find the lane/lane_connector ego belongs to
    for pose in poses[1:]:
        curr_route_obj = [one_route_obj for one_route_obj in curr_route_obj if one_route_obj.contains_point(pose)]

        if (not curr_route_obj) and len(route_objs[-1]):
            curr_route_obj = [
                next_route_obj
                for next_route_obj in route_objs[-1][0].outgoing_edges()
                if next_route_obj.contains_point(pose)
            ]
        if not curr_route_obj:
            curr_route_obj = get_curr_route_obj(map_api, pose)
        route_objs.append(curr_route_obj)

    # Iterate through route object and replace field with two lane_connectors with the one lane_connector ego ends up in
    for ind, curr_route_obj in enumerate(route_objs):
        if curr_route_obj:
            future_ind = ind + 1
            while future_ind < len(route_objs) - 1:
                if len(route_objs[future_ind]) == 1:
                    if route_objs[future_ind][0] in curr_route_obj:
                        route_objs[ind:future_ind] = [route_objs[future_ind]] * (future_ind - ind)
                    break
                future_ind += 1
            ind = future_ind
    return route_objs


def get_route_simplified(route_list: List[List[LaneGraphEdgeMapObject]]) -> List[List[LaneGraphEdgeMapObject]]:
    """
    This function simplifies the route by removing repeated consequtive route objects
    :param route_list: A list of route objects representing ego's corresponding route objects at each instance
    :return A simplified list of route objects that shows the order of route objects ego has been in.
    """
    # This function captures lane/lane_connector changes and does not necessarily return unique route objects
    assert len(route_list[0]), 'The first ego pose is not in a lane or lane_connector'
    route_simplified = [route_list[0]]
    for route_obj in route_list[1:]:
        if route_obj and route_obj != route_simplified[-1]:
            route_simplified.append(route_obj)
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

    for route_obj in expert_route:
        if route_obj:
            # simply get the first route obj
            roadblock_id = route_obj[0].get_roadblock_id()
            if roadblock_id != prev_roadblock_id:
                prev_roadblock_id = roadblock_id
                if isinstance(route_obj[0], Lane):
                    road_block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
                else:
                    road_block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)

                ref_baseline_path = route_obj[0].baseline_path()

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


def get_all_route_objs(map_api: AbstractMap, poses: List[Point2D]) -> List[List[GraphEdgeMapObject]]:
    """
    Returns and sets the sequence of lane and all lane connectors corresponding to the trajectory
    :param map_api: map
    :param poses: a list of xy coordinates
    :return list of route objects.
    """
    all_route_objs: List[List[GraphEdgeMapObject]] = []

    if not len(poses):
        logger.warning('Invalid poses passed to get_all_route_objs()')
        return all_route_objs

    # Find the lane/lane_connector ego belongs to initially
    curr_route_obj = get_curr_route_obj(map_api, poses[0])
    all_route_objs.append(curr_route_obj)

    # After finding the initial lane/lane_connector, for each pose first check if pose belongs to previously found
    # lane/lane_connector, if it does not check wether it's in an outgoing_egde of the previous lane/lane_connector,
    # otherwise find the lane/lane_connector ego belongs to
    if len(poses) > 1:
        for pose in poses[1:]:
            curr_route_obj = [one_route_obj for one_route_obj in curr_route_obj if one_route_obj.contains_point(pose)]
            if (not curr_route_obj) and len(all_route_objs[-1]):
                curr_route_obj = [
                    next_route_obj
                    for next_route_obj in all_route_objs[-1][0].outgoing_edges()
                    if next_route_obj.contains_point(pose)
                ]
            if not curr_route_obj:
                curr_route_obj = get_curr_route_obj(map_api, pose)
            all_route_objs.append(curr_route_obj)

    return all_route_objs


@dataclass
class CornersGraphEdgeMapObject:
    """class containing list of lane/lane connectors of each corner of ego's orientedbox."""

    front_left_map_objs: List[GraphEdgeMapObject]
    rear_left_map_objs: List[GraphEdgeMapObject]
    rear_right_map_objs: List[GraphEdgeMapObject]
    front_right_map_objs: List[GraphEdgeMapObject]


def extract_corners_route(map_api: AbstractMap, ego_corners: List[List[Point2D]]) -> List[CornersGraphEdgeMapObject]:
    """
    Extracts lists of lane/lane connectors of corners of ego from history
    :param map_api: AbstractMap
    :param ego_corners: List of x,y positions of corners of ego in the history (FL,RL,RR,FR)
    :return List of CornersGraphEdgeMapObject class containing list of lane/lane connectors of each
    corner of ego in the history.
    """
    front_left_route, rear_left_route, rear_right_route, front_right_route = [
        get_all_route_objs(map_api, corner_poses) for corner_poses in np.array(ego_corners).T.tolist()
    ]

    corners_route = []
    for (front_left_map_objs, rear_left_map_objs, rear_right_map_objs, front_right_map_objs) in zip(
        front_left_route, rear_left_route, rear_right_route, front_right_route
    ):

        corners_route.append(
            CornersGraphEdgeMapObject(
                front_left_map_objs, rear_left_map_objs, rear_right_map_objs, front_right_map_objs
            )
        )

    return corners_route


def get_outgoing_edges_obj_dict(corner_route_object: List[GraphEdgeMapObject]) -> dict[str, GraphEdgeMapObject]:
    """
    :param corner_route_object: List of lane/lane connectors
    :return dictionary of id and itscorresponding route object of outgoing edges of a given route object
    """
    return {obj_edge.id: obj_edge for obj in corner_route_object for obj_edge in obj.outgoing_edges()}


def get_incoming_edges_obj_dict(corner_route_object: List[GraphEdgeMapObject]) -> dict[str, GraphEdgeMapObject]:
    """
    :param corner_route_object: List of lane/lane connectors
    :return dictionary of id and itscorresponding route object of incoming edges of a given route object
    """
    return {obj_edge.id: obj_edge for obj in corner_route_object for obj_edge in obj.incoming_edges()}


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
    front_left_route_obj, rear_left_route_obj, rear_right_route_obj, front_right_route_obj = (
        corners_route_obj.front_left_map_objs,
        corners_route_obj.rear_left_map_objs,
        corners_route_obj.rear_right_map_objs,
        corners_route_obj.front_right_map_objs,
    )

    corners_route_obj_list = [front_left_route_obj, rear_left_route_obj, rear_right_route_obj, front_right_route_obj]

    in_nondrivable_area = [True if len(corner_route_obj) == 0 else False for corner_route_obj in corners_route_obj_list]

    # Check if all corners are in nondrivable area
    if np.all(in_nondrivable_area):
        return set()

    # Check if any corner (not all of them) is in nondrivable area
    if np.any(in_nondrivable_area):
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
    :param common_or_connected_route_objs: list of common or connected lane/lane connectors of corners if exist,
    empty list if all corners are in non_drivable area and None if corners are in different lane/lane connectors
    :param ego_timestamps: Array of times in time_us
    :return List of ego_timestamps where all corners of ego are in common or connected route objects
    """
    return [
        timestamp
        for route_obj, timestamp in zip(common_or_connected_route_objs, ego_timestamps)
        if route_obj is not None
    ]


def get_common_or_connected_route_objs_of_corners(
    corners_route: List[CornersGraphEdgeMapObject],
) -> List[Optional[Set[GraphEdgeMapObject]]]:
    """
    Extract timestamps when ego's corners are in different lane/lane connectors
    :param corners_route: List of class conatining list of lane/lane connectors of corners of ego
    :return list of common or connected lane/lane connectors of corners if exist, returns empty list if all corners are
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
