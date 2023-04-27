from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_acceleration import EgoYawAccelerationStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_yaw_acceleration/ego_yaw_acceleration.json')
def test_ego_yaw_acceleration(scene: Dict[str, Any]) -> None:
    """
    Tests ego yaw acceleration statistics as expected.
    :param scene: the json scene
    """
    metric = EgoYawAccelerationStatistics('ego_yaw_acceleration_statistics', 'Dynamics', max_abs_yaw_accel=3.0)
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
