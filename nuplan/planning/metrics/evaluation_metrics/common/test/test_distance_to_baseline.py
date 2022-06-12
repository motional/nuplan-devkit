from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.distance_to_baseline import DistanceToBaselineStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/distance_to_baseline/distance_to_baseline.json')
def test_distance_to_baseline(scene: Dict[str, Any]) -> None:
    """
    Tests drivable area violation metric, by checking the detection and the depth of violation on a made up scenario
    :param scene: the json scene.
    """
    metric = DistanceToBaselineStatistics('distance_to_baseline', 'Planning')
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
