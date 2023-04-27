from typing import Any, Dict

import pytest

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db
from nuplan.planning.metrics.utils.route_extractor import extract_corners_route, get_route, get_route_simplified

maps_db = get_test_maps_db()
map_factory = NuPlanMapFactory(maps_db)


@nuplan_test(path='json/route_extractor/route_extractor.json')
def test_get_route_and_simplify(scene: Dict[str, Any]) -> None:
    """
    Test getting route from ego pose and simplifying.
    """
    map_api = map_factory.build_map_from_name(scene['map']['area'])
    poses = []
    for marker in scene["markers"]:
        poses.append(Point2D(*marker["pose"][:2]))

    expert_route = get_route(map_api=map_api, poses=poses)
    assert len(expert_route) == len(poses)
    all_route_obj = [map_object for map_objects in expert_route for map_object in map_objects]
    assert len(all_route_obj) == len(poses)

    route_simplified = get_route_simplified(expert_route)
    assert len(route_simplified) == 3


@nuplan_test(path='json/route_extractor/route_extractor.json')
def test_corners_route_extraction(scene: Dict[str, Any]) -> None:
    """
    Test getting ego's corners route objects.
    """
    map_api = map_factory.build_map_from_name(scene['map']['area'])
    vehicle_parameters = get_pacifica_parameters()
    expert_footprints = []
    for marker in scene["markers"]:
        expert_footprints.append(CarFootprint.build_from_center(StateSE2(*marker["pose"][:3]), vehicle_parameters))
    corners_route = extract_corners_route(map_api=map_api, ego_footprint_list=expert_footprints)
    assert len(corners_route) == len(expert_footprints)
    all_route_obj = [
        map_object
        for corners_objects in corners_route
        for corner in corners_objects.__dict__.values()
        for map_object in corner
    ]
    unique_route_obj_ids = {obj.id for obj in all_route_obj}
    assert len(unique_route_obj_ids) == 4


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
