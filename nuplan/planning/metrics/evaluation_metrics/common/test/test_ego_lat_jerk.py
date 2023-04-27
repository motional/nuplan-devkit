from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_jerk import EgoLatJerkStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_lat_jerk/ego_lat_jerk.json')
def test_ego_lateral_jerk(scene: Dict[str, Any]) -> None:
    """
    Tests ego lateral jerk statistics as expected.
    :param scene: the json scene
    """
    metric = EgoLatJerkStatistics('ego_lat_jerk_statistics', 'Dynamics')
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
