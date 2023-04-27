from typing import Any, Callable, Dict, List

import pytest

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import Lane, PolylineMapObject
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.test_utils import add_map_objects_to_scene, add_polyline_to_scene
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db

maps_db = get_test_maps_db()
map_factory = NuPlanMapFactory(maps_db)


def assert_helper(
    first_markers: List[Dict[str, List[float]]],
    second_markers: List[Dict[str, List[float]]],
    assertion: Callable[[Lane, Lane, bool], None],
    map: AbstractMap,
    inverse: bool,
) -> None:
    """
    Helper function to remove redundant lane instantiation and checking
    """
    # first_markers and second_markers contains "pose": [float, float, float] which denotes marker location in the scene
    for first_marker, second_marker in zip(first_markers, second_markers):
        first_point = Point2D(*first_marker["pose"][:2])
        second_point = Point2D(*second_marker["pose"][:2])

        first_lane = map.get_one_map_object(first_point, SemanticMapLayer.LANE)
        second_lane = map.get_one_map_object(second_point, SemanticMapLayer.LANE)

        assertion(first_lane, second_lane, inverse)


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


@nuplan_test(path='json/lanes/lanes_in_same_roadblock.json')
def test_lane_is_same_roadblock(scene: Dict[str, Any]) -> None:
    """
    Test if lanes are in the same roadblock
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    def is_same_roadblock(first_lane: Lane, second_lane: Lane, inverse: bool) -> None:
        if not inverse:
            assert first_lane.is_same_roadblock(second_lane)
        else:
            assert not first_lane.is_same_roadblock(second_lane)

    # Index scheme that creates two lists containing every other marker over the specified range i.e. first_list = scene["markers"][i] and second_list = scene["markers"][i + 1]
    assert_helper(scene["markers"][:4:2], scene["markers"][1:4:2], is_same_roadblock, nuplan_map, False)
    assert_helper(scene["markers"][4::2], scene["markers"][5::2], is_same_roadblock, nuplan_map, True)


@nuplan_test(path='json/lanes/lanes_are_adjacent.json')
def test_lane_is_adjacent_to(scene: Dict[str, Any]) -> None:
    """
    Test if lanes are adjacent
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    def is_adjacent_to(first_lane: Lane, second_lane: Lane, inverse: bool) -> None:
        if not inverse:
            assert first_lane.is_adjacent_to(second_lane)
        else:
            assert not first_lane.is_adjacent_to(second_lane)

    # Index scheme that creates two lists containing every other marker over the specified range i.e. first_list = scene["markers"][i] and second_list = scene["markers"][i + 1]
    assert_helper(scene["markers"][:4:2], scene["markers"][1:4:2], is_adjacent_to, nuplan_map, False)
    assert_helper(scene["markers"][4::2], scene["markers"][5::2], is_adjacent_to, nuplan_map, True)


@nuplan_test(path='json/lanes/lane_is_left_of.json')
def test_lane_is_left_of(scene: Dict[str, Any]) -> None:
    """
    Test if first is left of second
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    def is_left_of(first_lane: Lane, second_lane: Lane, inverse: bool) -> None:
        if not inverse:
            assert first_lane.is_left_of(second_lane)
        else:
            assert not first_lane.is_left_of(second_lane)

    # Index scheme that creates two lists containing every other marker over the specified range i.e. first_list = scene["markers"][i] and second_list = scene["markers"][i + 1]
    assert_helper(scene["markers"][:4:2], scene["markers"][1:4:2], is_left_of, nuplan_map, False)
    assert_helper(scene["markers"][4::2], scene["markers"][5::2], is_left_of, nuplan_map, True)


@nuplan_test(path='json/lanes/lane_is_left_of.json')
def test_lane_is_right_of(scene: Dict[str, Any]) -> None:
    """
    Test if first is right of second
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    def is_right_of(first_lane: Lane, second_lane: Lane, inverse: bool) -> None:
        if not inverse:
            assert first_lane.is_right_of(second_lane)
        else:
            assert not first_lane.is_right_of(second_lane)

    # Index scheme that creates two lists containing every other marker over the specified range i.e. first_list = scene["markers"][i] and second_list = scene["markers"][i + 1]
    assert_helper(scene["markers"][1:4:2], scene["markers"][:4:2], is_right_of, nuplan_map, False)
    assert_helper(scene["markers"][5::2], scene["markers"][4::2], is_right_of, nuplan_map, True)


@nuplan_test(path='json/lanes/get_adjacent_lanes.json')
def test_get_lane_adjacent_lanes(scene: Dict[str, Any]) -> None:
    """
    Test if getting correct adjacent lanes
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])
    for marker in scene["markers"]:
        pose = marker["pose"]

        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        left_lane, right_lane = lane.adjacent_edges

        assert left_lane or right_lane
        if left_lane:
            assert left_lane.is_left_of(lane)
            assert left_lane.is_adjacent_to(lane)
        if right_lane:
            assert right_lane.is_right_of(lane)
            assert right_lane.is_adjacent_to(lane)


@nuplan_test(path='json/lanes/lane_index.json')
def test_get_lane_index(scene: Dict[str, Any]) -> None:
    """
    Test if getting correct lane index
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    # markers contains "pose": [float, float, float] which denotes marker location in the scene and expected_lane_index contains index of the lane at the location
    for marker, expected_index in zip(scene["markers"], scene["xtr"]["expected_lane_index"]):
        pose = marker["pose"]

        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is not None
        assert lane.index == expected_index


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
