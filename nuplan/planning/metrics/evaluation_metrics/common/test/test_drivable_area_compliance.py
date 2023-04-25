from typing import Any, Dict

import numpy as np
import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import DrivableAreaComplianceStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/drivable_area_compliance/drivable_area_violation.json')
def test_violations_detected_and_reported(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric, by checking the detection and the depth of violation on a made up scenario.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivableAreaComplianceStatistics('drivable_area_compliance', 'Planning', lane_change_metric, 0.3)
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/drivable_area_compliance/no_drivable_area_violation.json')
def test_works_with_no_violations(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric, by checking the detection and the depth of violation on a made up scenario.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivableAreaComplianceStatistics('drivable_area_compliance', 'Planning', lane_change_metric, 0.3)
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/drivable_area_compliance/small_drivable_area_violation.json')
def test_works_with_small_violations(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric when ego's footprint overapproximation is slightly outside drivable area.
    :param scene: the json scene
    """
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivableAreaComplianceStatistics('drivable_area_compliance', 'Planning', lane_change_metric, 0.3)
    metric_statistic_test(scene=scene, metric=metric)

    # Also check with a smaller threshold to make sure violation is catched:
    lane_change_metric = EgoLaneChangeStatistics('lane_change', 'Planning', 0.1)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    lane_change_metric.compute(history, mock_abstract_scenario)
    metric = DrivableAreaComplianceStatistics('drivable_area_compliance', 'Planning', lane_change_metric, 0.3)
    metric.compute(history, mock_abstract_scenario)
    assert np.isclose(metric.results[0].statistics[0].value, 0, atol=1e-2)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
