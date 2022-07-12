from typing import Any, Dict

import pytest

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import Lane, PolylineMapObject
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.test_utils import add_map_objects_to_scene, add_polyline_to_scene
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.nuplan_db_test_utils import get_test_maps_db

maps_db = get_test_maps_db()
map_factory = NuPlanMapFactory(maps_db)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_incoming_outgoing_lane_connectors(scene: Dict[str, Any]) -> None:
    """
    Test getting incoming and outgoing lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)

        assert lane is not None

        incoming_edges = lane.incoming_edges
        outgoing_edges = lane.outgoing_edges

        assert len(incoming_edges) > 0
        assert len(outgoing_edges) > 0

        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)


@nuplan_test(path='json/connections/no_end_connection.json')
def test_no_end_lane_connector(scene: Dict[str, Any]) -> None:
    """
    Test when there are not outgoing lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)

        assert lane is not None

        incoming_edges = lane.incoming_edges
        outgoing_edges = lane.outgoing_edges

        assert not outgoing_edges

        add_map_objects_to_scene(scene, incoming_edges)


@nuplan_test(path='json/connections/no_start_connection.json')
def test_no_start_lane_connector(scene: Dict[str, Any]) -> None:
    """
    Test when there are not incoming lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)

        assert lane is not None

        incoming_edges = lane.incoming_edges
        outgoing_edges = lane.outgoing_edges

        assert not incoming_edges

        add_map_objects_to_scene(scene, outgoing_edges)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_lane_left_boundaries(scene: Dict[str, Any]) -> None:
    """
    Test getting left boundaries of lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)

        assert lane is not None

        left_boundary = lane.left_boundary

        assert left_boundary is not None
        assert isinstance(left_boundary, PolylineMapObject)

        add_polyline_to_scene(scene, left_boundary.discrete_path)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_lane_right_boundaries(scene: Dict[str, Any]) -> None:
    """
    Test getting right boundaries of lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)

        assert lane is not None

        right_boundary = lane.right_boundary

        assert right_boundary is not None
        assert isinstance(right_boundary, PolylineMapObject)

        add_polyline_to_scene(scene, right_boundary.discrete_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
