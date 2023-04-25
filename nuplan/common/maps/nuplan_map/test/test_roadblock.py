from typing import Any, Dict

import pytest
from shapely.geometry import Polygon

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.test_utils import add_map_objects_to_scene
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db

maps_db = get_test_maps_db()
map_factory = NuPlanMapFactory(maps_db)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_incoming_outgoing_roadblock_connectors(scene: Dict[str, Any]) -> None:
    """
    Test getting incoming and outgoing roadblock connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK
        )

        assert roadblock is not None

        incoming_edges = roadblock.incoming_edges
        outgoing_edges = roadblock.outgoing_edges

        assert len(incoming_edges) > 0
        assert len(outgoing_edges) > 0

        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)


@nuplan_test(path='json/connections/no_end_connection.json')
def test_no_end_roadblock_connector(scene: Dict[str, Any]) -> None:
    """
    Test when there are not outgoing roadblock connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK
        )

        assert roadblock is not None

        incoming_edges = roadblock.incoming_edges
        outgoing_edges = roadblock.outgoing_edges

        assert not outgoing_edges

        add_map_objects_to_scene(scene, incoming_edges)


@nuplan_test(path='json/connections/no_start_connection.json')
def test_no_start_roadblock_connector(scene: Dict[str, Any]) -> None:
    """
    Test when there are not incoming lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK
        )

        assert roadblock is not None

        incoming_edges = roadblock.incoming_edges
        outgoing_edges = roadblock.outgoing_edges

        assert not incoming_edges

        add_map_objects_to_scene(scene, outgoing_edges)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_roadblock_interior_edges(scene: Dict[str, Any]) -> None:
    """
    Test getting roadblock's interior lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK
        )

        assert roadblock is not None

        interior_edges = roadblock.interior_edges

        assert len(interior_edges) > 0

        add_map_objects_to_scene(scene, interior_edges)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_roadblock_polygon(scene: Dict[str, Any]) -> None:
    """
    Test getting roadblock's polygon.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock: RoadBlockGraphEdgeMapObject = nuplan_map.get_one_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK
        )

        assert roadblock is not None

        polygon = roadblock.polygon

        assert polygon
        assert isinstance(polygon, Polygon)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
