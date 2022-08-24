from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.driving_direction_violation import (
    DrivingDirectionViolationStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/driving_direction_violation/ego_does_not_drive_backward.json')
def test_ego_no_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when there's no route.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivingDirectionViolationStatistics(
        "driving_direction_violation",
        "Planning",
        lane_change_metric,
        driving_direction_violation_threshold=2,
        time_horizon=1,
    )
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/driving_direction_violation/ego_drives_backward.json')
def test_ego_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when ego drives backward.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivingDirectionViolationStatistics(
        "driving_direction_violation",
        "Planning",
        lane_change_metric,
        driving_direction_violation_threshold=2,
        time_horizon=1,
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
