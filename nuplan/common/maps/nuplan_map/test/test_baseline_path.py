from typing import Any, Dict

import pytest

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.maps.nuplan_map.polyline_map_object import NuPlanPolylineMapObject
from nuplan.common.maps.nuplan_map.utils import get_row_with_value
from nuplan.common.maps.test_utils import add_map_objects_to_scene, add_marker_to_scene
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db

maps_db = get_test_maps_db()
map_factory = NuPlanMapFactory(maps_db)


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

    for marker, exp_arc_length, exp_pose, exp_curv in zip(
        scene["markers"], expected_arc_length, expected_pose.values(), expected_curvature
    ):
        pose = marker["pose"]
        point = Point2D(pose[0], pose[1])
        lane = nuplan_map.get_one_map_object(point, SemanticMapLayer.LANE)

        assert lane is not None
        assert lane.contains_point(point)

        add_map_objects_to_scene(scene, [lane])
        lane_blp = lane.baseline_path

        arc_length = lane_blp.get_nearest_arc_length_from_position(point)
        pose = lane_blp.get_nearest_pose_from_position(point)
        curv = lane_blp.get_curvature_at_arc_length(arc_length)

        poses[marker["id"]] = pose

        assert arc_length == pytest.approx(exp_arc_length)
        assert pose == StateSE2(exp_pose[0], exp_pose[1], exp_pose[2])
        assert curv == pytest.approx(exp_curv)

        # test baseline_path() calls constructor as expected
        constructed_blp = NuPlanPolylineMapObject(get_row_with_value(lane._baseline_paths_df, "lane_fid", lane.id))
        constructed_blp_arc_length = constructed_blp.get_nearest_arc_length_from_position(point)
        constructed_blp_pose = constructed_blp.get_nearest_pose_from_position(point)
        constructed_blp_curv = constructed_blp.get_curvature_at_arc_length(constructed_blp_arc_length)

        assert arc_length == pytest.approx(constructed_blp_arc_length)
        assert pose == constructed_blp_pose
        assert curv == pytest.approx(constructed_blp_curv)

    for pose_id, pose in poses.items():
        add_marker_to_scene(scene, str(pose_id), pose)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
