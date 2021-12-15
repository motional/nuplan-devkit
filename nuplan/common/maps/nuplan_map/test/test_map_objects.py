import os
from typing import Any, Dict, List

import pytest
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import Crosswalk, Intersection, Lane, LaneConnector, StopLine
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.test_utils import add_map_objects_to_scene, add_marker_to_scene
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

map_factory = NuPlanMapFactory(GPKGMapsDB('nuplan-maps-v0.1',
                                          map_root=os.path.join(os.getenv('NUPLAN_DATA_ROOT', "~/nuplan/dataset"),
                                                                'maps')))


@nuplan_test(path='json/baseline/baseline_in_lane.json')
def test_baseline_queries_in_lane(scene: Dict[str, Any]) -> None:
    """
    Test baseline queries.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])
    expected_arc_length = scene["xtr"]["expected_arc_length"]
    expected_pose = scene["xtr"]["expected_pose"]
    expected_curvature = scene["xtr"]["expected_curvature"]

    poses = {}
    for marker, exp_arc_length, exp_pose, exp_curv \
            in zip(scene["markers"], expected_arc_length, expected_pose.values(), expected_curvature):
        pose = marker["pose"]
        point = Point2D(pose[0], pose[1])
        lane = nuplan_map.get_one_map_object(point, SemanticMapLayer.LANE)

        assert lane is not None
        assert lane.contains_point(point)

        add_map_objects_to_scene(scene, [lane])
        lane_blp = lane.baseline_path()

        arc_length = lane_blp.get_nearest_arc_length_from_position(point)
        pose = lane_blp.get_nearest_pose_from_position(point)
        curv = lane_blp.get_curvature_at_arc_length(arc_length)

        poses[marker["id"]] = pose

        assert arc_length == pytest.approx(exp_arc_length)
        assert pose == StateSE2(exp_pose[0], exp_pose[1], exp_pose[2])
        assert curv == pytest.approx(exp_curv)

    for pose_id, pose in poses.items():
        add_marker_to_scene(scene, str(pose_id), pose)


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

        incoming_edges = lane.incoming_edges()
        outgoing_edges = lane.outgoing_edges()

        assert len(incoming_edges) > 0
        assert len(outgoing_edges) > 0

        add_map_objects_to_scene(scene, incoming_edges)
        add_map_objects_to_scene(scene, outgoing_edges)


@nuplan_test(path='json/baseline/baseline_in_intersection.json')
def test_get_incoming_outgoing_lanes(scene: Dict[str, Any]) -> None:
    """
    Test getting incoming and outgoing lanes.
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker in scene["markers"]:
        pose = marker["pose"]

        lane_connectors: List[LaneConnector] = nuplan_map.get_all_map_objects(Point2D(pose[0], pose[1]),
                                                                              SemanticMapLayer.LANE_CONNECTOR)
        assert len(lane_connectors) > 0

        incoming_edges = lane_connectors[0].incoming_edges()
        outgoing_edges = lane_connectors[0].outgoing_edges()

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

        incoming_edges = lane.incoming_edges()
        outgoing_edges = lane.outgoing_edges()

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

        incoming_edges = lane.incoming_edges()
        outgoing_edges = lane.outgoing_edges()

        assert not incoming_edges

        add_map_objects_to_scene(scene, outgoing_edges)


@nuplan_test(path='json/stop_lines/nearby.json')
def test_get_nearby_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test getting nearby stop lines
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_distance, expected_id in zip(scene["markers"],
                                                      scene["xtr"]["expected_nearest_distance"],
                                                      scene["xtr"]["expected_nearest_id"]):
        pose = marker["pose"]

        stop_line_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]),
                                                                               SemanticMapLayer.STOP_LINE)
        assert stop_line_id is not None
        assert expected_distance == distance
        assert expected_id == stop_line_id

        stop_line: StopLine = nuplan_map.get_map_object(stop_line_id, SemanticMapLayer.STOP_LINE)
        add_map_objects_to_scene(scene, [stop_line])


@nuplan_test(path='json/stop_lines/on_stopline.json')
def test_get_stop_lines(scene: Dict[str, Any]) -> None:
    """
    Test getting stop lines at a point
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_id in zip(scene["markers"], scene["xtr"]["expected_nearest_id"]):
        pose = marker["pose"]

        stop_line: StopLine = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.STOP_LINE)

        assert stop_line is not None
        assert expected_id == stop_line.id
        assert stop_line.contains_point(Point2D(pose[0], pose[1]))

        add_map_objects_to_scene(scene, [stop_line])


@nuplan_test(path='json/crosswalks/nearby.json')
def test_get_nearby_crosswalks(scene: Dict[str, Any]) -> None:
    """
    Test getting nearby crosswalks
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_distance, expected_id in zip(scene["markers"],
                                                      scene["xtr"]["expected_nearest_distance"],
                                                      scene["xtr"]["expected_nearest_id"]):
        pose = marker["pose"]

        crosswalk_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]),
                                                                               SemanticMapLayer.CROSSWALK)

        assert crosswalk_id is not None
        assert expected_distance == distance
        assert expected_id == crosswalk_id

        crosswalk: Crosswalk = nuplan_map.get_map_object(crosswalk_id, SemanticMapLayer.CROSSWALK)
        add_map_objects_to_scene(scene, [crosswalk])


@nuplan_test(path='json/crosswalks/on_crosswalk.json')
def test_get_crosswalk(scene: Dict[str, Any]) -> None:
    """
    Test getting crosswalk at a point
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_id in zip(scene["markers"], scene["xtr"]["expected_nearest_id"]):
        pose = marker["pose"]

        crosswalk: Crosswalk = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]), SemanticMapLayer.CROSSWALK)

        assert crosswalk is not None
        assert expected_id == crosswalk.id
        assert crosswalk.contains_point(Point2D(pose[0], pose[1]))

        add_map_objects_to_scene(scene, [crosswalk])


@nuplan_test(path='json/intersections/on_intersection.json')
def test_get_intersections(scene: Dict[str, Any]) -> None:
    """
    Test getting intersections at a point
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_id in zip(scene["markers"], scene["xtr"]["expected_nearest_id"]):
        pose = marker["pose"]

        intersection: Intersection = nuplan_map.get_one_map_object(Point2D(pose[0], pose[1]),
                                                                   SemanticMapLayer.INTERSECTION)
        assert intersection is not None
        assert expected_id == intersection.id
        assert intersection.contains_point(Point2D(pose[0], pose[1]))

        add_map_objects_to_scene(scene, [intersection])


@nuplan_test(path='json/intersections/nearby.json')
def test_get_nearby_intersection(scene: Dict[str, Any]) -> None:
    """
    Test getting nearby crosswalks
    """
    nuplan_map = map_factory.build_map_from_name(scene["map"]["area"])

    for marker, expected_distance, expected_id in zip(scene["markers"],
                                                      scene["xtr"]["expected_nearest_distance"],
                                                      scene["xtr"]["expected_nearest_id"]):
        pose = marker["pose"]

        intersection_id, distance = nuplan_map.get_distance_to_nearest_map_object(Point2D(pose[0], pose[1]),
                                                                                  SemanticMapLayer.INTERSECTION)

        assert intersection_id is not None
        assert expected_distance == distance
        assert expected_id == intersection_id

        intersection: Intersection = nuplan_map.get_map_object(intersection_id, SemanticMapLayer.INTERSECTION)
        add_map_objects_to_scene(scene, [intersection])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
