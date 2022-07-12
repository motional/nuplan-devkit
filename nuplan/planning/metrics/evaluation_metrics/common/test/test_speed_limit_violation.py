from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.speed_limit_violation import SpeedLimitViolationStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/speed_limit_violation/speed_limit_violation.json')
def test_speed_limit_violation(scene: Dict[str, Any]) -> None:
    """
    Tests speed limit violation metric, by checking the detection and the depth of violation on a made up scenario
    :param scene: the json scene.
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = SpeedLimitViolationStatistics(
        'speed_limit_violation',
        '',
        lane_change_metric=lane_change_metric,
        max_violation_threshold=1,
        max_overspeed_value_threshold=2.23,
    )
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/speed_limit_violation/speed_limit_no_violation.json')
def test_no_violations(scene: Dict[str, Any]) -> None:
    """
    Tests speed limit violation metric, by checking that the metric works without violations
    :param scene: the json scene.
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = SpeedLimitViolationStatistics(
        'speed_limit_violation',
        '',
        lane_change_metric=lane_change_metric,
        max_violation_threshold=1,
        max_overspeed_value_threshold=2.23,
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
