from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_expert_l2_error import EgoExpertL2ErrorStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_expert_l2_error/ego_expert_l2_error.json')
def test_ego_expert_l2_error(scene: Dict[str, Any]) -> None:
    """
    Tests ego expert l2 error is expected value.
    :param scene: the json scene
    """
    metric = EgoExpertL2ErrorStatistics('ego_expert_L2_error', 'Dynamics', discount_factor=1.0)
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
