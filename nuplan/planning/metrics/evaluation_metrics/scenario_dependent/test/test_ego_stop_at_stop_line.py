from typing import Any, Dict, Optional

import pytest
from shapely.geometry import LineString

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.scenario_dependent.ego_stop_at_stop_line import (
    EgoStopAtStopLineStatistics,
)
from nuplan.planning.metrics.metric_result import TimeSeries
from nuplan.planning.metrics.utils.testing_utils import setup_history
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario


@nuplan_test(path='json/ego_stop_at_stop_line/ego_stop_at_stop_line.json')
def test_stop_polygons_in_lanes(scene: Dict[str, Any]) -> None:
    """
    Check if verification of stop polygons in lanes works as expected
    :param scene: the json scene.
    """
    mock_abstract_scenario = MockAbstractScenario()
    history = setup_history(scene, scenario=mock_abstract_scenario)
    ego_stop_at_stop_line_metric = EgoStopAtStopLineStatistics(
        name='ego_stop_at_stop_line',
        category='scenario_dependent',
        distance_threshold=5.0,
        velocity_threshold=0.1,
        max_violation_threshold=1,
    )
    map_api: AbstractMap = history.map_api
    valid_stop_polygons = []
    for data in history.data:

        # Get egos' front footprint
        ego_corners = data.ego_state.car_footprint.oriented_box.geometry.exterior.coords  # [FL, RL, RR, FR]
        ego_pose_front: LineString = LineString([ego_corners[0], ego_corners[3]])

        stop_polygon_info = ego_stop_at_stop_line_metric.get_nearest_stop_line(
            map_api=map_api, ego_pose_front=ego_pose_front
        )
        if stop_polygon_info is not None:
            valid_stop_polygons.append(stop_polygon_info)

    assert len(history.data) == 6
    assert len(valid_stop_polygons) == 6


@nuplan_test(path='json/ego_stop_at_stop_line/ego_stop_at_stop_line.json')
def test_check_leading_agent(scene: Dict[str, Any]) -> None:
    """
    Check if check_leading_agent work as expected
    :param scene: the json scene.
    """
    mock_abstract_scenario = MockAbstractScenario()
    history = setup_history(scene, scenario=mock_abstract_scenario)
    ego_stop_at_stop_line_metric = EgoStopAtStopLineStatistics(
        name='ego_stop_at_stop_line',
        category='scenario_dependent',
        distance_threshold=5.0,
        velocity_threshold=0.1,
        max_violation_threshold=1,
    )
    map_api: AbstractMap = history.map_api

    # Set True to remove all agents to test check_for_leading_agents works as expected
    remove_agents = [False, False, False, True, True, False]
    expected_results = [True, True, True, False, False, False]
    results = []

    for data, remove_agent in zip(history.data, remove_agents):
        detections = data.observation
        if remove_agent:
            detections.boxes = []
        has_leading_agent = ego_stop_at_stop_line_metric.check_for_leading_agents(
            detections=detections, ego_state=data.ego_state, map_api=map_api
        )
        results.append(has_leading_agent)
    assert expected_results == results


@nuplan_test(path='json/ego_stop_at_stop_line/ego_stop_at_stop_line.json')
def test_egos_stop_at_stop_line(scene: Dict[str, Any]) -> None:
    """
    Check if egos stop at stop line as expected
    :param scene: the json scene.
    """
    # Remove all detections
    scene['world']['vehicles'] = []
    mock_abstract_scenario = MockAbstractScenario()
    history = setup_history(scene, scenario=mock_abstract_scenario)
    ego_stop_at_stop_line_metric = EgoStopAtStopLineStatistics(
        name='ego_stop_at_stop_line',
        category='scenario_dependent',
        distance_threshold=5.0,
        velocity_threshold=0.1,
        max_violation_threshold=1,
    )
    results = ego_stop_at_stop_line_metric.compute(history=history, scenario=mock_abstract_scenario)
    assert len(results) == 1

    result = results[0]
    metric_statistics = result.statistics
    time_series: Optional[TimeSeries] = result.time_series

    assert metric_statistics[0].value == 1
    assert metric_statistics[1].value == 1
    assert metric_statistics[2].value == 0.06016734670118855
    assert metric_statistics[3].value == 0.05

    expected_velocity = [0.5, 0.05]
    assert time_series.values if time_series is not None else [] == expected_velocity


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
