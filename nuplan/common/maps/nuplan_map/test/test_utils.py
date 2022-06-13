from typing import Any, Dict, List

import pytest

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map_objects import Lane, LaneConnector, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import (
    LaneOnRouteStatusData,
    LaneSegmentConnections,
    LaneSegmentCoords,
    LaneSegmentGroupings,
    LaneSegmentLaneIDs,
    LaneSegmentRoadBlockIDs,
    OnRouteStatusType,
    SemanticMapLayer,
)
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.nuplan_map.utils import (
    build_lane_segments_from_blps,
    connect_blp_lane_segments,
    connect_lane_conn_predecessor,
    connect_lane_conn_successor,
    extract_roadblock_objects,
    get_neighbor_vector_map,
    get_on_route_status,
    get_roadblock_ids_from_trajectory,
    group_blp_lane_segments,
    split_blp_lane_segments,
)
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.nuplan_db_test_utils import get_test_maps_db
from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario

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
    Test grouping lane indices beloning to same lane/lane connector.
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
def test_build_lane_segments_from_blps(scene: Dict[str, Any]) -> None:
    """
    Test splitting baseline paths into lane segments.
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

        incoming_edges = lane_connector.incoming_edges()

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

        outgoing_edges = lane_connector.outgoing_edges()

        assert len(outgoing_edges) > 0

        cross_blp_conns: Dict[str, List[int]] = {}
        cross_blp_conns[lane_connector.id] = [0, 0]
        cross_blp_conns[outgoing_edges[0].id] = [0, 0]

        lane_seg_suc_conns = connect_lane_conn_successor(lane_connector, cross_blp_conns)
        assert len(lane_seg_suc_conns) > 0
        assert isinstance(lane_seg_suc_conns, List)
        assert isinstance(lane_seg_suc_conns[0], tuple)
        assert isinstance(lane_seg_suc_conns[0][0], int)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_neighbor_vector_map(scene: Dict[str, Any]) -> None:
    """
    Test constructing lane segment info from given map api.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        radius = 20

        (
            lane_seg_coords,
            lane_seg_conns,
            lane_seg_groupings,
            lane_seg_lane_ids,
            lane_seg_roadblock_ids,
        ) = get_neighbor_vector_map(nuplan_map, Point2D(pose[0], pose[1]), radius)

        assert type(lane_seg_coords) == LaneSegmentCoords
        assert type(lane_seg_conns) == LaneSegmentConnections
        assert type(lane_seg_groupings) == LaneSegmentGroupings
        assert type(lane_seg_lane_ids) == LaneSegmentLaneIDs
        assert type(lane_seg_roadblock_ids) == LaneSegmentRoadBlockIDs


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_extract_roadblock_objects(scene: Dict[str, Any]) -> None:
    """
    Test extract roadblock or roadblock connectors from map containing point.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock_objects = extract_roadblock_objects(nuplan_map, Point2D(pose[0], pose[1]))

        assert isinstance(roadblock_objects, List)

        for roadblock_object in roadblock_objects:
            assert isinstance(roadblock_object, RoadBlockGraphEdgeMapObject)


def test_get_roadblock_ids_from_trajectory() -> None:
    """
    Test extracting ids of roadblocks and roadblock connectors containing points specified in trajectory.
    """
    scenario = get_test_nuplan_scenario()

    roadblock_ids = get_roadblock_ids_from_trajectory(scenario.map_api, scenario.get_expert_ego_trajectory())

    assert isinstance(roadblock_ids, List)

    for roadblock_id in roadblock_ids:
        assert isinstance(roadblock_id, str)


def test_get_on_route_status() -> None:
    """
    Test identifying whether given roadblock lie within goal route.
    """
    route_roadblock_ids = ["0"]
    roadblock_ids = LaneSegmentRoadBlockIDs(["0", "1"])

    on_route_status = get_on_route_status(route_roadblock_ids, roadblock_ids)

    assert type(on_route_status) == LaneOnRouteStatusData
    assert len(on_route_status.on_route_status) == 2
    assert on_route_status.on_route_status[0] == on_route_status.encode(OnRouteStatusType.ON_ROUTE)
    assert on_route_status.on_route_status[1] == on_route_status.encode(OnRouteStatusType.OFF_ROUTE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
