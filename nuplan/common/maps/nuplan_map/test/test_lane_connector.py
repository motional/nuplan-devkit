from typing import Any, Dict, List

import pytest
from shapely.geometry import Point

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import LaneConnector, PolylineMapObject
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.test_utils import add_map_objects_to_scene, add_polyline_to_scene
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db

maps_db = get_test_maps_db()
map_factory = NuPlanMapFactory(maps_db)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_incoming_outgoing_lanes(scene: Dict[str, Any]) -> None:
    """
    Test getting incoming and outgoing lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )
        assert len(lane_connectors) > 0

        incoming_edges = lane_connectors[0].incoming_edges
        outgoing_edges = lane_connectors[0].outgoing_edges

        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_lane_left_boundaries(scene: Dict[str, Any]) -> None:
    """
    Test getting left boundaries of lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )
        assert len(lane_connectors) > 0

        left_boundary = lane_connectors[0].left_boundary

        assert left_boundary is not None
        assert isinstance(left_boundary, PolylineMapObject)

        add_polyline_to_scene(scene, left_boundary.discrete_path)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_lane_right_boundaries(scene: Dict[str, Any]) -> None:
    """
    Test getting right boundaries of lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )
        assert len(lane_connectors) > 0

        right_boundary = lane_connectors[0].right_boundary

        assert right_boundary is not None
        assert isinstance(right_boundary, PolylineMapObject)

        add_polyline_to_scene(scene, right_boundary.discrete_path)


@nuplan_test(path='json/intersections/on_intersection_with_stop_lines.json')
def test_get_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test getting stop lines from lane connector.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )
        assert len(lane_connectors) > 0

        stop_lines = lane_connectors[0].stop_lines

        assert len(stop_lines) > 0

        add_map_objects_to_scene(scene, stop_lines)


@nuplan_test(path='json/intersections/on_intersection_with_no_stop_lines.json')
def test_get_stop_lines_empty(scene: Dict[str, Any]) -> None:
    """
    Test getting stop lines from lane connector when there are no stop lines.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )
        assert len(lane_connectors) > 0

        stop_lines = lane_connectors[0].stop_lines

        assert len(stop_lines) == 0

        add_map_objects_to_scene(scene, stop_lines)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_polygon(scene: Dict[str, Any]) -> None:
    """
    Test getting polygons from lane_connector.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        point = Point(pose[0], pose[1])

        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )
        assert len(lane_connectors) > 0

        polygon = lane_connectors[0].polygon
        assert polygon.contains(point)

        add_map_objects_to_scene(scene, lane_connectors)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
