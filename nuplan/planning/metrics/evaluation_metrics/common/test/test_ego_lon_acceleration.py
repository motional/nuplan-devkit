from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration import EgoLonAccelerationStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_lon_acceleration/ego_lon_acceleration.json')
def test_ego_longitudinal_acceleration(scene: Dict[str, Any]) -> None:
    """
    Tests ego longitudinal acceleration statistics as expected
    :param scene: the json scene.
    """
    metric = EgoLonAccelerationStatistics(
        'ego_lon_acceleration_statistics', 'Dynamics', min_lon_accel=0.0, max_lon_accel=10.0
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
