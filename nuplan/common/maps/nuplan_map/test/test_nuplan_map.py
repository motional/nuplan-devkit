from typing import Any, Dict

import numpy as np
import pytest

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import Lane
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.test_utils import add_map_objects_to_scene
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db


@pytest.fixture()
def map_factory() -> NuPlanMapFactory:
    """Fixture loading ta returning a map factory"""
    maps_db = get_test_maps_db()
    return NuPlanMapFactory(maps_db)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_is_in_layer_lane(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test is in lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        assert nuplan_map.is_in_layer(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_is_in_layer_intersection(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test is in intersection.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        assert nuplan_map.is_in_layer(Point2D(pose[0], pose[1]), SemanticMapLayer.INTERSECTION)


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_lane(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
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
def test_no_baseline(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
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
def test_get_lane_connector(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
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

    pose = scene["markers"][0]["pose"]

    with pytest.raises(AssertionError):
        nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR)


@nuplan_test(path='json/get_nearest/lane.json')
def test_get_nearest_lane(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting nearest lane.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_distance, expected_id in zip(
        scene["markers"], scene["xtr"]["expected_nearest_distance"], scene["xtr"]["expected_nearest_id"]
    ):
        pose = marker["pose"]
        lane_id, distance = nuplan_map.get_distance_to_nearest_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE
        )

        assert lane_id == expected_id
        assert distance == expected_distance

        lane = nuplan_map.get_map_object(str(lane_id), SemanticMapLayer.LANE)

        add_map_objects_to_scene(scene, [lane])


@nuplan_test(path='json/get_nearest/lane_connector.json')
def test_get_nearest_lane_connector(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting nearest lane connector.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_distance, expected_id in zip(
        scene["markers"], scene["xtr"]["expected_nearest_distance"], scene["xtr"]["expected_nearest_id"]
    ):
        pose = marker["pose"]
        lane_connector_id, distance = nuplan_map.get_distance_to_nearest_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.LANE_CONNECTOR
        )
        lane_connector = nuplan_map.get_map_object(str(lane_connector_id), SemanticMapLayer.LANE_CONNECTOR)

        add_map_objects_to_scene(scene, [lane_connector])


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_get_roadblock(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting one roadblock.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        point = Point2D(pose[0], pose[1])
        roadblock = nuplan_map.get_one_map_object(point, SemanticMapLayer.ROADBLOCK)

        assert roadblock is not None
        assert roadblock.contains_point(point)

        add_map_objects_to_scene(scene, [roadblock])


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_roadblock_connector(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting roadblock connectors.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        point = Point2D(pose[0], pose[1])

        roadblock_connectors = nuplan_map.get_all_map_objects(point, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        assert roadblock_connectors is not None

        add_map_objects_to_scene(scene, roadblock_connectors)
        for roadblock_connector in roadblock_connectors:
            assert roadblock_connector.contains_point(point)

    pose = scene["markers"][0]["pose"]

    with pytest.raises(AssertionError):
        nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR)


@nuplan_test(path='json/get_nearest/lane.json')
def test_get_nearest_roadblock(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting nearest roadblock.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        roadblock_id, distance = nuplan_map.get_distance_to_nearest_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK
        )

        roadblock = nuplan_map.get_map_object(str(roadblock_id), SemanticMapLayer.ROADBLOCK)

        assert roadblock_id

        add_map_objects_to_scene(scene, [roadblock])


@nuplan_test(path='json/get_nearest/lane_connector.json')
def test_get_nearest_roadblock_connector(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test getting nearest roadblock connector.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]
        roadblock_connector_id, distance = nuplan_map.get_distance_to_nearest_map_object(
            Point2D(pose[0], pose[1]), SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        assert roadblock_connector_id != -1
        assert distance != np.NaN

        roadblock_connector = nuplan_map.get_map_object(
            str(roadblock_connector_id), SemanticMapLayer.ROADBLOCK_CONNECTOR
        )

        assert roadblock_connector

        add_map_objects_to_scene(scene, [roadblock_connector])


@nuplan_test(path='json/neighboring/all_map_objects.json')
def test_get_proximal_map_objects(scene: Dict[str, Any], map_factory: NuPlanMapFactory) -> None:
    """
    Test get_neighbor_lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    marker = scene["markers"][0]
    pose = marker["pose"]
    map_objects = nuplan_map.get_proximal_map_objects(
        Point2D(pose[0], pose[1]),
        40,
        [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.ROADBLOCK_CONNECTOR,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
        ],
    )

    assert len(map_objects[SemanticMapLayer.LANE]) == scene["xtr"]["expected_num_lanes"]
    assert len(map_objects[SemanticMapLayer.LANE_CONNECTOR]) == scene["xtr"]["expected_num_lane_connectors"]
    assert len(map_objects[SemanticMapLayer.ROADBLOCK]) == scene["xtr"]["expected_num_roadblocks"]
    assert len(map_objects[SemanticMapLayer.ROADBLOCK_CONNECTOR]) == scene["xtr"]["expected_num_roadblock_connectors"]
    assert len(map_objects[SemanticMapLayer.STOP_LINE]) == scene["xtr"]["expected_num_stop_lines"]
    assert len(map_objects[SemanticMapLayer.CROSSWALK]) == scene["xtr"]["expected_num_cross_walks"]
    assert len(map_objects[SemanticMapLayer.INTERSECTION]) == scene["xtr"]["expected_num_intersections"]

    for layer, map_objects in map_objects.items():
        add_map_objects_to_scene(scene, map_objects, layer)


@nuplan_test()
def test_unsupported_neighbor_map_objects(map_factory: NuPlanMapFactory) -> None:
    """
    Test throw if unsupported layer is queried.
    """
    nuplan_map = map_factory.build_map_from_name("us-nv-las-vegas-strip")

    with pytest.raises(AssertionError):
        nuplan_map.get_proximal_map_objects(
            Point2D(0, 0),
            15,
            [
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.ROADBLOCK_CONNECTOR,
                SemanticMapLayer.STOP_LINE,
                SemanticMapLayer.CROSSWALK,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.TRAFFIC_LIGHT,
            ],
        )


@nuplan_test()
def test_get_available_map_objects(map_factory: NuPlanMapFactory) -> None:
    """
    Test getting available map objects for all SemanticMapLayers.
    """
    nuplan_map = map_factory.build_map_from_name("us-nv-las-vegas-strip")

    assert set(nuplan_map.get_available_map_objects()) == {
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
    }


def test_get_drivable_area(map_factory: NuPlanMapFactory) -> None:
    """Tests drivable area construction"""
    nuplan_map = map_factory.build_map_from_name("us-nv-las-vegas-strip")

    target_layer = "drivable_area"
    base_layers = ["road_segments", "intersections", "generic_drivable_areas", "carpark_areas"]
    all_layers = base_layers + [target_layer]
    assert not any(layer in nuplan_map._vector_map.keys() for layer in all_layers)
    nuplan_map._load_vector_map_layer(target_layer)
    assert all(layer in nuplan_map._vector_map.keys() for layer in all_layers)
    drivable_fids = nuplan_map._vector_map[target_layer]['fid'].to_list()
    base_fids = [fid for layer in base_layers for fid in nuplan_map._vector_map[layer]['fid'].to_list()]
    assert sorted(drivable_fids) == sorted(base_fids)


def test_initialize_all_layers(map_factory: NuPlanMapFactory) -> None:
    """Tests initialize all layers function"""
    nuplan_map = map_factory.build_map_from_name("us-nv-las-vegas-strip")

    assert not nuplan_map._vector_map
    nuplan_map.initialize_all_layers()
    assert nuplan_map._vector_map


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
