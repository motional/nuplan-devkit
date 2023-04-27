from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_acceleration import EgoAccelerationStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_acceleration/ego_acceleration.json')
def test_ego_expected_acceleration(scene: Dict[str, Any]) -> None:
    """
    Tests ego acceleration by checking if it is the expected acceleration.
    :param scene: the json scene
    """
    metric = EgoAccelerationStatistics("ego_acceleration_statistics", "Dynamics")
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
