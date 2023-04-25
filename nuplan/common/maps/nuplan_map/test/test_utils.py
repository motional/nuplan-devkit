from typing import Any, Dict, List

import pytest

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.test.test_utils import get_sample_ego_state
from nuplan.common.maps.abstract_map_objects import Lane, LaneConnector, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.nuplan_map.utils import (
    build_lane_segments_from_blps,
    build_lane_segments_from_blps_with_trim,
    connect_blp_lane_segments,
    connect_lane_conn_predecessor,
    connect_lane_conn_successor,
    connect_trimmed_lane_conn_predecessor,
    connect_trimmed_lane_conn_successor,
    extract_polygon_from_map_object,
    extract_roadblock_objects,
    get_distance_between_map_object_and_point,
    get_roadblock_ids_from_trajectory,
    group_blp_lane_segments,
    split_blp_lane_segments,
)
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db

maps_db = get_test_maps_db()
map_factory = NuPlanMapFactory(maps_db)


def test_split_blp_lane_segments() -> None:
    """
    Test splitting baseline paths node list into lane segments.
    """
    nodes = [StateSE2(0.0, 0.0, 0.0), StateSE2(0.0, 0.0, 0.0), StateSE2(0.0, 0.0, 0.0)]
    lane_seg_num = 2
    obj_coords = split_blp_lane_segments(nodes, lane_seg_num)

    assert len(obj_coords) == 2
    assert len(obj_coords[0]) == 2
    assert len(obj_coords[0][0]) == 2

    assert isinstance(obj_coords, List)
    assert isinstance(obj_coords[0], List)
    assert isinstance(obj_coords[0][0], List)
    assert isinstance(obj_coords[0][0][0], float)


def test_connect_blp_lane_segments() -> None:
    """
    Test connecting lane indices.
    """
    start_lane_seg_idx = 0
    lane_seg_num = 10
    obj_conns = connect_blp_lane_segments(start_lane_seg_idx, lane_seg_num)

    assert len(obj_conns) == lane_seg_num - 1
    assert len(obj_conns[0]) == 2

    assert isinstance(obj_conns, List)
    assert isinstance(obj_conns[0], tuple)
    assert isinstance(obj_conns[0][0], int)


def test_group_blp_lane_segments() -> None:
    """
    Test grouping lane indices belonging to same lane/lane connector.
    """
    start_lane_seg_idx = 0
    lane_seg_num = 10
    obj_groupings = group_blp_lane_segments(start_lane_seg_idx, lane_seg_num)

    assert len(obj_groupings) == 1
    assert len(obj_groupings[0]) == lane_seg_num

    assert isinstance(obj_groupings, List)
    assert isinstance(obj_groupings[0], List)
    assert isinstance(obj_groupings[0][0], int)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_build_lane_segments_from_blps_with_trim(scene: Dict[str, Any]) -> None:
    """
    Test build and trim the lane segments from the baseline paths.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        radius = 20

        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)

        assert lane is not None

        start_idx = 0
        (
            trimmed_obj_coords,
            trimmed_obj_conns,
            trimmed_obj_groupings,
            trimmed_obj_lane_ids,
            trimmed_obj_roadblock_ids,
            trimmed_obj_cross_blp_conn,
        ) = build_lane_segments_from_blps_with_trim(Point2D(pose[0], pose[1]), radius, lane, start_idx)

        start_idx = 0
        (
            obj_coords,
            obj_conns,
            obj_groupings,
            obj_lane_ids,
            obj_roadblock_ids,
            obj_cross_blp_conn,
        ) = build_lane_segments_from_blps(lane, start_idx)

        assert len(trimmed_obj_coords) > 0
        assert len(trimmed_obj_conns) > 0
        assert len(trimmed_obj_groupings) > 0
        assert len(trimmed_obj_lane_ids) > 0
        assert len(trimmed_obj_roadblock_ids) > 0
        assert len(trimmed_obj_cross_blp_conn) == 2

        assert len(trimmed_obj_coords) == len(trimmed_obj_conns) + 1
        assert len(trimmed_obj_coords) == len(trimmed_obj_groupings[0])
        assert len(trimmed_obj_coords) == len(trimmed_obj_lane_ids)
        assert len(trimmed_obj_coords) == len(trimmed_obj_roadblock_ids)

        assert len(trimmed_obj_coords) <= len(obj_coords)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_connect_trimmed_lane_conn_predecessor(scene: Dict[str, Any]) -> None:
    """
    Test connecting trimmed lane connector to incoming lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        lane_connector: LaneConnector = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )[0]
        assert lane_connector is not None
        incoming_edges = lane_connector.incoming_edges
        assert len(incoming_edges) > 0

        lane: Lane = lane_connector.incoming_edges[0]
        assert lane is not None
        start_idx = 0
        radius = 20
        trim_nodes = build_lane_segments_from_blps_with_trim(Point2D(pose[0], pose[1]), radius, lane, start_idx)
        if trim_nodes is not None:
            (
                obj_coords,
                obj_conns,
                obj_groupings,
                obj_lane_ids,
                obj_roadblock_ids,
                obj_cross_blp_conn,
            ) = trim_nodes
        else:
            continue

        cross_blp_conns: Dict[str, List[int]] = {}
        cross_blp_conns[lane_connector.id] = [0, 0]
        cross_blp_conns[incoming_edges[0].id] = [0, 0]

        lane_seg_pred_conns = connect_trimmed_lane_conn_predecessor(obj_coords, lane_connector, cross_blp_conns)
        assert len(lane_seg_pred_conns) > 0
        assert isinstance(lane_seg_pred_conns, List)
        assert isinstance(lane_seg_pred_conns[0], tuple)
        assert isinstance(lane_seg_pred_conns[0][0], int)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_connect_trimmed_lane_conn_successor(scene: Dict[str, Any]) -> None:
    """
    Test connecting trimmed lane connector to outgoing lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        lane_connector: LaneConnector = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )[0]
        assert lane_connector is not None
        outgoing_edges = lane_connector.outgoing_edges
        assert len(outgoing_edges) > 0

        lane: Lane = lane_connector.outgoing_edges[0]
        assert lane is not None
        start_idx = 0
        radius = 20
        trim_nodes = build_lane_segments_from_blps_with_trim(Point2D(pose[0], pose[1]), radius, lane, start_idx)
        if trim_nodes is not None:
            (
                obj_coords,
                obj_conns,
                obj_groupings,
                obj_lane_ids,
                obj_roadblock_ids,
                obj_cross_blp_conn,
            ) = trim_nodes
        else:
            continue

        cross_blp_conns: Dict[str, List[int]] = {}
        cross_blp_conns[lane_connector.id] = [0, 0]
        cross_blp_conns[outgoing_edges[0].id] = [0, 0]

        lane_seg_suc_conns = connect_trimmed_lane_conn_successor(obj_coords, lane_connector, cross_blp_conns)
        assert len(lane_seg_suc_conns) > 0
        assert isinstance(lane_seg_suc_conns, List)
        assert isinstance(lane_seg_suc_conns[0], tuple)
        assert isinstance(lane_seg_suc_conns[0][0], int)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_build_lane_segments_from_blps(scene: Dict[str, Any]) -> None:
    """
    Test building lane segments from baseline paths.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)

        assert lane is not None

        start_idx = 0
        (
            obj_coords,
            obj_conns,
            obj_groupings,
            obj_lane_ids,
            obj_roadblock_ids,
            obj_cross_blp_conn,
        ) = build_lane_segments_from_blps(lane, start_idx)

        assert len(obj_coords) > 0
        assert len(obj_conns) > 0
        assert len(obj_groupings) > 0
        assert len(obj_lane_ids) > 0
        assert len(obj_roadblock_ids) > 0
        assert len(obj_cross_blp_conn) == 2

        assert len(obj_coords) == len(obj_conns) + 1
        assert len(obj_coords) == len(obj_groupings[0])
        assert len(obj_coords) == len(obj_lane_ids)
        assert len(obj_coords) == len(obj_roadblock_ids)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_connect_lane_conn_predecessor(scene: Dict[str, Any]) -> None:
    """
    Test connecting lane connector to incoming lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        lane_connector: LaneConnector = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )[0]

        assert lane_connector is not None

        incoming_edges = lane_connector.incoming_edges

        assert len(incoming_edges) > 0

        cross_blp_conns: Dict[str, List[int]] = {}
        cross_blp_conns[lane_connector.id] = [0, 0]
        cross_blp_conns[incoming_edges[0].id] = [0, 0]

        lane_seg_pred_conns = connect_lane_conn_predecessor(lane_connector, cross_blp_conns)
        assert len(lane_seg_pred_conns) > 0
        assert isinstance(lane_seg_pred_conns, List)
        assert isinstance(lane_seg_pred_conns[0], tuple)
        assert isinstance(lane_seg_pred_conns[0][0], int)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_connect_lane_conn_successor(scene: Dict[str, Any]) -> None:
    """
    Test connecting lane connector to outgoing lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        lane_connector: LaneConnector = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )[0]

        assert lane_connector is not None

        outgoing_edges = lane_connector.outgoing_edges

        assert len(outgoing_edges) > 0

        cross_blp_conns: Dict[str, List[int]] = {}
        cross_blp_conns[lane_connector.id] = [0, 0]
        cross_blp_conns[outgoing_edges[0].id] = [0, 0]

        lane_seg_suc_conns = connect_lane_conn_successor(lane_connector, cross_blp_conns)
        assert len(lane_seg_suc_conns) > 0
        assert isinstance(lane_seg_suc_conns, List)
        assert isinstance(lane_seg_suc_conns[0], tuple)
        assert isinstance(lane_seg_suc_conns[0][0], int)


@nuplan_test(path='json/crosswalks/nearby.json')
def test_extract_polygon_from_map_object_crosswalk(scene: Dict[str, Any]) -> None:
    """
    Test extracting polygon from map object. Tests crosswalks.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])
    radius = 20

    for marker in scene["markers"]:
        pose = marker["pose"]

        layers = nuplan_map.get_proximal_map_objects(Point2D(pose[0], pose[1]), radius, [SemanticMapLayer.CROSSWALK])
        crosswalks = layers[SemanticMapLayer.CROSSWALK]

        assert len(crosswalks) > 0

        crosswalk_polygon = extract_polygon_from_map_object(crosswalks[0])

        assert isinstance(crosswalk_polygon, List)
        assert len(crosswalk_polygon) > 0
        assert isinstance(crosswalk_polygon[0], Point2D)


@nuplan_test(path='json/stop_lines/nearby.json')
def test_extract_polygon_from_map_object_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test extracting polygon from map object. Tests stop lines.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])
    radius = 20

    for marker in scene["markers"]:
        pose = marker["pose"]

        layers = nuplan_map.get_proximal_map_objects(Point2D(pose[0], pose[1]), radius, [SemanticMapLayer.STOP_LINE])
        stop_lines = layers[SemanticMapLayer.STOP_LINE]

        assert len(stop_lines) > 0

        stop_line_polygon = extract_polygon_from_map_object(stop_lines[0])

        assert isinstance(stop_line_polygon, List)
        assert len(stop_line_polygon) > 0
        assert isinstance(stop_line_polygon[0], Point2D)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_extract_roadblock_objects_roadblocks(scene: Dict[str, Any]) -> None:
    """
    Test extract roadblock or roadblock connectors from map containing point. Tests roadblocks.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock_objects = extract_roadblock_objects(nuplan_map, Point2D(pose[0], pose[1]))

        assert isinstance(roadblock_objects, List)
        assert len(roadblock_objects) > 0

        roadblock_object = roadblock_objects[0]
        assert isinstance(roadblock_object, RoadBlockGraphEdgeMapObject)

        roadblock_polygon = extract_polygon_from_map_object(roadblock_object)

        assert isinstance(roadblock_polygon, List)
        assert len(roadblock_polygon) > 0
        assert isinstance(roadblock_polygon[0], Point2D)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_extract_roadblock_objects_roadblock_connectors(scene: Dict[str, Any]) -> None:
    """
    Test extract roadblock or roadblock connectors from map containing point. Tests roadblock connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock_objects = extract_roadblock_objects(nuplan_map, Point2D(pose[0], pose[1]))

        assert isinstance(roadblock_objects, List)
        assert len(roadblock_objects) > 0

        roadblock_object = roadblock_objects[0]
        assert isinstance(roadblock_object, RoadBlockGraphEdgeMapObject)

        roadblock_polygon = extract_polygon_from_map_object(roadblock_object)

        assert isinstance(roadblock_polygon, List)
        assert len(roadblock_polygon) > 0
        assert isinstance(roadblock_polygon[0], Point2D)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_roadblock_ids_from_trajectory(scene: Dict[str, Any]) -> None:
    """
    Test extracting ids of roadblocks and roadblock connectors containing points specified in trajectory.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    # Trajectory to extract route from
    trajectory: List[EgoState] = []
    for marker in scene["markers"]:
        pose = marker["pose"]
        ego_state = get_sample_ego_state()
        ego_state.car_footprint.rear_axle = StateSE2(pose[0], pose[1], pose[2])
        trajectory.append(ego_state)

    roadblock_ids = get_roadblock_ids_from_trajectory(nuplan_map, trajectory)

    assert isinstance(roadblock_ids, List)

    for roadblock_id in roadblock_ids:
        assert isinstance(roadblock_id, str)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_distance_between_map_object_and_point_lanes_roadblocks(scene: Dict[str, Any]) -> None:
    """
    Test get distance between point and nearest surface of specified map object.
    Tests lane/connectors and roadblock/connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])
    radius = 35
    pose = scene["markers"][0]["pose"]
    point = Point2D(pose[0], pose[1])

    layer_names = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
    ]
    layers = nuplan_map.get_proximal_map_objects(point, radius, layer_names)
    for layer_name in layer_names:
        map_objects = layers[layer_name]
        assert len(map_objects) > 0
        dist = get_distance_between_map_object_and_point(point, map_objects[0])
        assert dist <= radius


@nuplan_test(path='json/crosswalks/nearby.json')
def test_get_distance_between_map_object_and_point_crosswalks(scene: Dict[str, Any]) -> None:
    """
    Test get distance between point and nearest surface of specified map object. Tests crosswalks.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])
    radius = 35
    pose = scene["markers"][0]["pose"]
    point = Point2D(pose[0], pose[1])
    layers = nuplan_map.get_proximal_map_objects(point, radius, [SemanticMapLayer.CROSSWALK])
    map_objects = layers[SemanticMapLayer.CROSSWALK]
    assert len(map_objects) > 0
    dist = get_distance_between_map_object_and_point(point, map_objects[0])
    assert dist <= radius


@nuplan_test(path='json/stop_lines/nearby.json')
def test_get_distance_between_map_object_and_point_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test get distance between point and nearest surface of specified map object. Tests stop lines.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])
    radius = 35
    pose = scene["markers"][0]["pose"]
    point = Point2D(pose[0], pose[1])

    layers = nuplan_map.get_proximal_map_objects(point, radius, [SemanticMapLayer.STOP_LINE])
    map_objects = layers[SemanticMapLayer.STOP_LINE]
    assert len(map_objects) > 0
    dist = get_distance_between_map_object_and_point(point, map_objects[0])
    assert dist <= radius


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
