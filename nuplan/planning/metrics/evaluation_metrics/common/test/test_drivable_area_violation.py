from typing import Any, Dict

import pytest
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_violation import DrivableAreaViolationStatistics
from nuplan.planning.metrics.metric_result import MetricStatisticsType
from nuplan.planning.metrics.utils.testing_utils import setup_history


@nuplan_test(path='json/drivable_area_violation/drivable_area_violation.json')
def test_violations_detected_and_reported(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric, by checking the detection and the depth of violation on a made up scenario.

    :param scene: the json scene
    """
    history = setup_history(scene)

    metric = DrivableAreaViolationStatistics("drivable_area_violation", "")
    drivable_area_violation_statistics = metric.compute(history)[0]
    statistics = drivable_area_violation_statistics.statistics
    assert statistics[MetricStatisticsType.COUNT].value == 2
    assert round(statistics[MetricStatisticsType.MAX].value, 2) == 1.47
    assert round(statistics[MetricStatisticsType.MEAN].value, 2) == 1.28


@nuplan_test(path='json/drivable_area_violation/drivable_area_violation.json')
def test_works_with_no_violations(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric, by checking the detection and the depth of violation on a made up scenario.

    :param scene: the json scene
    """
    history = setup_history(scene)
    history.data = history.data[:2]
    metric = DrivableAreaViolationStatistics("drivable_area_violation", "")
    drivable_area_violation_statistics = metric.compute(history)[0]

    assert drivable_area_violation_statistics.statistics[MetricStatisticsType.COUNT].value == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
