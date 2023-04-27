from typing import Any, Dict, List

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


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_incoming_outgoing_roadblock(scene: Dict[str, Any]) -> None:
    """
    Test getting incoming and outgoing lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock_connectors: List[RoadBlockGraphEdgeMapObject] = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        assert len(roadblock_connectors) > 0

        incoming_edges = roadblock_connectors[0].incoming_edges
        outgoing_edges = roadblock_connectors[0].outgoing_edges

        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_roadblock_connector_interior_edges(scene: Dict[str, Any]) -> None:
    """
    Test getting roadblock connector's interior lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock_connectors: List[RoadBlockGraphEdgeMapObject] = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR
        )

        assert len(roadblock_connectors) > 0

        interior_edges = roadblock_connectors[0].interior_edges

        assert len(interior_edges) > 0

        add_map_objects_to_scene(scene, interior_edges)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_roadblock_connector_polygon(scene: Dict[str, Any]) -> None:
    """
    Test getting roadblock connector's polygon.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        roadblock_connectors: List[RoadBlockGraphEdgeMapObject] = nuplan_map.get_all_map_objects(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR
        )

        assert len(roadblock_connectors) > 0

        polygon = roadblock_connectors[0].polygon

        assert polygon
        assert isinstance(polygon, Polygon)


if __name__ == "__main__":
    import os

    os.unsetenv('PYTEST_PLUGINS')
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
