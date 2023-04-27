from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_mean_speed import EgoMeanSpeedStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_mean_speed/ego_mean_speed.json')
def test_ego_mean_speed(scene: Dict[str, Any]) -> None:
    """
    Tests ego mean speed statistics as expected.
    :param scene: the json scene
    """
    metric = EgoMeanSpeedStatistics('ego_lon_jerk_statistics', 'Dynamics')
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
