from typing import Any, Dict, Optional

import pytest
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.metrics.evaluation_metrics.scenario_dependent.ego_stop_at_stop_line import \
    EgoStopAtStopLineStatistics
from nuplan.planning.metrics.metric_result import MetricStatisticsType, TimeSeries
from nuplan.planning.metrics.utils.testing_utils import setup_history
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import ego_state_to_box_3d
from shapely.geometry import LineString


@nuplan_test(path='json/ego_stop_at_stop_line/ego_stop_at_stop_line.json')
def test_stop_polygons_in_lanes(scene: Dict[str, Any]) -> None:
    """
    Check if verification of stop polygons in lanes works as expected.
    :param scene: the json scene.
    """
    history = setup_history(scene)
    ego_stop_at_stop_line_metric = EgoStopAtStopLineStatistics(name='ego_stop_at_stop_line',
                                                               category='scenario_dependent',
                                                               distance_threshold=5.0,
                                                               velocity_threshold=0.1)
    map_api: AbstractMap = history.map_api
    valid_stop_polygons = []
    for data in history.data:
        ego_pose_box_3d: Box3D = ego_state_to_box_3d(data.ego_state)

        # Get egos' front footprint
        ego_pose_front: LineString = LineString(ego_pose_box_3d.bottom_corners[:2, :2].T)

        stop_polygon_info = ego_stop_at_stop_line_metric.get_nearest_stop_line(
            map_api=map_api,
            ego_pose_front=ego_pose_front
        )
        if stop_polygon_info is not None:
            valid_stop_polygons.append(stop_polygon_info)

    assert len(history.data) == 6
    assert len(valid_stop_polygons) == 6


@nuplan_test(path='json/ego_stop_at_stop_line/ego_stop_at_stop_line.json')
def test_check_leading_agent(scene: Dict[str, Any]) -> None:
    """
    Check if check_leading_agent work as expected.
    :param scene: the json scene.
    """

    history = setup_history(scene)
    ego_stop_at_stop_line_metric = EgoStopAtStopLineStatistics(name='ego_stop_at_stop_line',
                                                               category='scenario_dependent',
                                                               distance_threshold=5.0,
                                                               velocity_threshold=0.1)
    map_api: AbstractMap = history.map_api

    # Set True to remove all agents to test check_for_leading_agents works as expected
    remove_agents = [False, False, False, True, True, False]
    expected_results = [True, True, True, False, False, False]
    results = []

    for data, remove_agent in zip(history.data, remove_agents):
        ego_pose_box_3d: Box3D = ego_state_to_box_3d(data.ego_state)
        detections = data.observation
        if remove_agent:
            detections.boxes = []
        has_leading_agent = ego_stop_at_stop_line_metric.check_for_leading_agents(
            detections=detections,
            ego_box_3d=ego_pose_box_3d,
            map_api=map_api
        )
        results.append(has_leading_agent)
    assert expected_results == results


@nuplan_test(path='json/ego_stop_at_stop_line/ego_stop_at_stop_line.json')
def test_egos_stop_at_stop_line(scene: Dict[str, Any]) -> None:
    """
    Check if egos stop at stop line as expected.
    :param scene: the json scene.
    """

    # Remove all detections
    scene['world']['vehicles'] = []
    history = setup_history(scene)
    ego_stop_at_stop_line_metric = EgoStopAtStopLineStatistics(name='ego_stop_at_stop_line',
                                                               category='scenario_dependent',
                                                               distance_threshold=5.0,
                                                               velocity_threshold=0.1)
    results = ego_stop_at_stop_line_metric.compute(history=history)
    assert len(results) == 1

    result = results[0]
    metric_statistics = result.statistics
    time_series: Optional[TimeSeries] = result.time_series

    assert metric_statistics[MetricStatisticsType.DISTANCE].value == 0.06016734670118855
    assert metric_statistics[MetricStatisticsType.VELOCITY].value == 0.05
    assert metric_statistics[MetricStatisticsType.BOOLEAN].value is True

    expected_velocity = [0.5, 0.05]
    assert time_series.values if time_series is not None else [] == expected_velocity


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
