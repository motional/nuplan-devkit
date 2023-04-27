from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_acceleration import EgoLatAccelerationStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_lat_acceleration/ego_lat_acceleration.json')
def test_ego_lateral_acceleration(scene: Dict[str, Any]) -> None:
    """
    Tests ego lateral acceleration statistics as expected.
    :param scene: the json scene
    """
    metric = EgoLatAccelerationStatistics('ego_lat_acceleration_statistics', 'Dynamics', max_abs_lat_accel=10.0)
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
