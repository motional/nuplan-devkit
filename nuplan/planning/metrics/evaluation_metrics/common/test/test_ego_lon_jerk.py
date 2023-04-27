from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_jerk import EgoLonJerkStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_lon_jerk/ego_lon_jerk.json')
def test_ego_longitudinal_jerk(scene: Dict[str, Any]) -> None:
    """
    Tests ego longitudinal jerk statistics as expected.
    :param scene: the json scene
    """
    metric = EgoLonJerkStatistics('ego_lon_jerk_statistics', 'Dynamics', max_abs_lon_jerk=8.0)
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
