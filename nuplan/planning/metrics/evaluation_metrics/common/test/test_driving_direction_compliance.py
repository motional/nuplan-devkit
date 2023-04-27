from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.driving_direction_compliance import (
    DrivingDirectionComplianceStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/driving_direction_compliance/ego_does_not_drive_backward.json')
def test_ego_no_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when there's no route.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivingDirectionComplianceStatistics(
        "driving_direction_compliance",
        "Planning",
        lane_change_metric,
        driving_direction_compliance_threshold=2,
        driving_direction_violation_threshold=6,
        time_horizon=1,
    )
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/driving_direction_compliance/ego_drives_backward.json')
def test_ego_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when ego drives backward more than driving_direction_violation_threshold.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivingDirectionComplianceStatistics(
        "driving_direction_compliance",
        "Planning",
        lane_change_metric,
        driving_direction_compliance_threshold=2,
        driving_direction_violation_threshold=6,
        time_horizon=1,
    )
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/driving_direction_compliance/ego_slightly_drives_backward.json')
def test_ego_slightly_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when ego drives backward more than driving_direction_compliance_threshold but less than driving_direction_violation_threshold.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivingDirectionComplianceStatistics(
        "driving_direction_compliance",
        "Planning",
        lane_change_metric,
        driving_direction_compliance_threshold=2,
        driving_direction_violation_threshold=15,
        time_horizon=1,
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
