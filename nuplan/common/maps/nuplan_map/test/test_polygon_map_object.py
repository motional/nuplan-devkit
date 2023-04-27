from typing import Any, Dict

import pytest

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import PolygonMapObject
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.test_utils import add_map_objects_to_scene
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db

maps_db = get_test_maps_db()
map_factory = NuPlanMapFactory(maps_db)


@nuplan_test(path='json/crosswalks/nearby.json')
def test_get_nearby_crosswalks(scene: Dict[str, Any]) -> None:
    """
    Test getting nearby crosswalks.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_distance, expected_id in zip(
        scene["markers"], scene["xtr"]["expected_nearest_distance"], scene["xtr"]["expected_nearest_id"]
    ):
        pose = marker["pose"]

        crosswalk_id, distance = nuplan_map.get_distance_to_nearest_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.CROSSWALK
        )

        assert crosswalk_id is not None
        assert expected_distance == distance
        assert expected_id == crosswalk_id

        crosswalk: PolygonMapObject = nuplan_map.get_map_object(crosswalk_id, SemanticMapLayer.CROSSWALK)
        add_map_objects_to_scene(scene, [crosswalk])


@nuplan_test(path='json/crosswalks/on_crosswalk.json')
def test_get_crosswalk(scene: Dict[str, Any]) -> None:
    """
    Test getting crosswalk at a point.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_id in zip(scene["markers"], scene["xtr"]["expected_nearest_id"]):
        pose = marker["pose"]

        crosswalk: PolygonMapObject = nuplan_map.get_one_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.CROSSWALK
        )

        assert crosswalk is not None
        assert expected_id == crosswalk.id
        assert crosswalk.contains_point(Point2D(pose[0], pose[1]))

        add_map_objects_to_scene(scene, [crosswalk])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
