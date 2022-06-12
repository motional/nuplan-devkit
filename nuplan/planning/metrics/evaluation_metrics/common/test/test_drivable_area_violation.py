from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_violation import DrivableAreaViolationStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/drivable_area_violation/drivable_area_violation.json')
def test_violations_detected_and_reported(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric, by checking the detection and the depth of violation on a made up scenario.
    :param scene: the json scene
    """
    metric = DrivableAreaViolationStatistics('drivable_area_violation', 'Violations', 1)
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/drivable_area_violation/drivable_area_no_violation.json')
def test_works_with_no_violations(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric, by checking the detection and the depth of violation on a made up scenario.
    :param scene: the json scene
    """
    metric = DrivableAreaViolationStatistics('drivable_area_violation', 'Violations', 1)
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
