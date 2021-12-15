import os
from typing import Any, Dict

import pytest
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import Lane
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.test_utils import add_map_objects_to_scene
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

map_factory = NuPlanMapFactory(
    GPKGMapsDB('nuplan-maps-v0.1', map_root=os.path.join(os.getenv('NUPLAN_DATA_ROOT', "~/nuplan/dataset"), 'maps')))


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_is_in_layer_lane(scene: Dict[str, Any]) -> None:
    """
    Test is in lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        assert nuplan_map.is_in_layer(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_is_in_layer_intersection(scene: Dict[str, Any]) -> None:
    """
    Test is in intersection.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        assert nuplan_map.is_in_layer(Point2D(pose[0], pose[1]), SemanticMapLayer.INTERSECTION)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_lane(scene: Dict[str, Any]) -> None:
    """
    Test getting one lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_speed_limit in zip(scene["markers"], scene["xtr"]["expected_speed_limit"]):
        pose = marker["pose"]
        point = Point2D(pose[0], pose[1])
        lane = nuplan_map.get_one_map_object(point, SemanticMapLayer.LANE)

        assert lane is not None
        assert lane.contains_point(point)
        assert lane.speed_limit_mps == pytest.approx(expected_speed_limit)

        add_map_objects_to_scene(scene, [lane])


@nuplan_test(path='json/baseline/no_baseline.json')
def test_no_baseline(scene: Dict[str, Any]) -> None:
    """
    Test when there is no baseline.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        lane: Lane = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)
        assert lane is None
        lane_connector = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)
        assert not lane_connector


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_lane_connector(scene: Dict[str, Any]) -> None:
    """
    Test getting lane connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    idx = 0
    for marker in scene["markers"]:
        pose = marker["pose"]
        point = Point2D(pose[0], pose[1])

        lane_connectors = nuplan_map.get_all_map_objects(point, SemanticMapLayer.LANE_CONNECTOR)
        assert lane_connectors is not None

        add_map_objects_to_scene(scene, lane_connectors)
        for lane_connector in lane_connectors:
            assert lane_connector.contains_point(point)
            assert lane_connector.speed_limit_mps == pytest.approx(scene["xtr"]["expected_speed_limit"][idx])
            idx += 1

    with pytest.raises(AssertionError):
        pose = scene["markers"][0]["pose"]
        nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)


@nuplan_test(path='json/get_nearest/lane.json')
def test_get_nearest_lane(scene: Dict[str, Any]) -> None:
    """
    Test getting nearest lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_distance, expected_id in zip(scene["markers"],
                                                      scene["xtr"]["expected_nearest_distance"],
                                                      scene["xtr"]["expected_nearest_id"]):
        pose = marker["pose"]
        lane_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]),
                                                                          SemanticMapLayer.LANE)

        assert lane_id == expected_id
        assert distance == expected_distance

        lane = nuplan_map.get_map_object(str(lane_id), SemanticMapLayer.LANE)

        add_map_objects_to_scene(scene, [lane])


@nuplan_test(path='json/get_nearest/lane_connector.json')
def test_get_nearest_lane_connector(scene: Dict[str, Any]) -> None:
    """
    Test getting nearest lane connector.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_distance, expected_id in zip(scene["markers"],
                                                      scene["xtr"]["expected_nearest_distance"],
                                                      scene["xtr"]["expected_nearest_id"]):
        pose = marker["pose"]
        lane_connector_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]),
                                                                                    SemanticMapLayer.LANE_CONNECTOR)
        # TODO: restore checks
        # assert lane_connector_id is not -1
        # assert distance is not np.NaN

        lane_connector = nuplan_map.get_map_object(str(lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)

        add_map_objects_to_scene(scene, [lane_connector])


@nuplan_test(path='json/neighboring/all_map_objects.json')
def test_get_proximal_map_objects(scene: Dict[str, Any]) -> None:
    """
    Test get_neighbor_lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    marker = scene["markers"][0]
    pose = marker["pose"]
    map_objects = nuplan_map.get_proximal_map_objects(Point2D(pose[0], pose[1]), 40, [SemanticMapLayer.LANE,
                                                                                      SemanticMapLayer.LANE_CONNECTOR,
                                                                                      SemanticMapLayer.STOP_LINE,
                                                                                      SemanticMapLayer.CROSSWALK,
                                                                                      SemanticMapLayer.INTERSECTION])

    assert len(map_objects[SemanticMapLayer.LANE]) == scene["xtr"]["expected_num_lanes"]
    assert len(map_objects[SemanticMapLayer.LANE_CONNECTOR]) == scene["xtr"]["expected_num_lane_connectors"]
    assert len(map_objects[SemanticMapLayer.STOP_LINE]) == scene["xtr"]["expected_num_stop_lines"]
    assert len(map_objects[SemanticMapLayer.CROSSWALK]) == scene["xtr"]["expected_num_cross_walks"]
    assert len(map_objects[SemanticMapLayer.INTERSECTION]) == scene["xtr"]["expected_num_intersections"]

    for layer, map_objects in map_objects.items():
        add_map_objects_to_scene(scene, map_objects, layer)


@nuplan_test()
def test_unsupported_neighbor_map_objects() -> None:
    """
    Test throw if unsupported layer is queried
    """
    nuplan_map = map_factory.build_map_from_name("us-nv-las-vegas-strip")

    with pytest.raises(ValueError):
        nuplan_map.get_proximal_map_objects(Point2D(0, 0), 15, [SemanticMapLayer.LANE,
                                                                SemanticMapLayer.LANE_CONNECTOR,
                                                                SemanticMapLayer.STOP_LINE,
                                                                SemanticMapLayer.CROSSWALK,
                                                                SemanticMapLayer.INTERSECTION,
                                                                SemanticMapLayer.TRAFFIC_LIGHT])


@nuplan_test()
def test_get_available_map_objects() -> None:
    nuplan_map = map_factory.build_map_from_name("us-nv-las-vegas-strip")

    assert set(nuplan_map.get_available_map_objects()) == {SemanticMapLayer.LANE,
                                                           SemanticMapLayer.LANE_CONNECTOR,
                                                           SemanticMapLayer.STOP_LINE,
                                                           SemanticMapLayer.CROSSWALK,
                                                           SemanticMapLayer.INTERSECTION}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
