from typing import Any, Dict

import pytest
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.speed_limit_violation import SpeedLimitViolationStatistics
from nuplan.planning.metrics.metric_result import MetricStatisticsType
from nuplan.planning.metrics.utils.testing_utils import setup_history


@nuplan_test(path='json/speed_limit_violation/speed_limit_violation_lane.json')
def test_violations_detected_and_reported_lane(scene: Dict[str, Any]) -> None:
    """
    Tests speed limit violation metric, by checking the detection and the depth of violation on a made up scenario.

    :param scene: the json scene
    """
    history = setup_history(scene)

    metric = SpeedLimitViolationStatistics("drivable_area_violation", "")
    speed_limit_violation_statistics = metric.compute(history)[0]

    statistics = speed_limit_violation_statistics.statistics
    assert statistics[MetricStatisticsType.COUNT].value == 1, "Wrong number of violations detected"
    assert round(statistics[MetricStatisticsType.MAX].value, 2) == 21.86, "Wrong maximal violation"
    assert round(statistics[MetricStatisticsType.MEAN].value, 2) == 13.22, "Wrong mean violation"


@nuplan_test(path='json/speed_limit_violation/speed_limit_violation_lane_connector.json')
def test_works_with_no_violations(scene: Dict[str, Any]) -> None:
    """
    Tests speed limit violation metric, by checking that the metric works without violations.

    :param scene: the json scene
    """
    history = setup_history(scene)
    history.data = history.data[-1:]

    metric = SpeedLimitViolationStatistics("drivable_area_violation", "")
    speed_limit_violation_statistics = metric.compute(history)[0]
    assert speed_limit_violation_statistics.statistics[MetricStatisticsType.COUNT].value == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
