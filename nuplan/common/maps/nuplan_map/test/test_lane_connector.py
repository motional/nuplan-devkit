from typing import Any, Dict, List

import pytest

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import LaneConnector
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.test_utils import add_map_objects_to_scene
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.nuplan_db_test_utils import get_test_maps_db

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

        incoming_edges = lane_connectors[0].incoming_edges()
        outgoing_edges = lane_connectors[0].outgoing_edges()

        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)


@nuplan_test(path='json/intersections/on_intersection_with_stop_lines.json')
def test_get_stop_lines(scene: Dict[str, Any]) -> None:
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

        stop_lines = lane_connectors[0].get_stop_lines()

        assert len(stop_lines) > 0

        add_map_objects_to_scene(scene, stop_lines)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
