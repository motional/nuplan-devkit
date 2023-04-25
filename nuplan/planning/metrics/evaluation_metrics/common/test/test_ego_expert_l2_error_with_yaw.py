from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_expert_l2_error_with_yaw import (
    EgoExpertL2ErrorWithYawStatistics,
)
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_expert_l2_error_with_yaw/ego_expert_l2_error_with_yaw.json')
def test_ego_expert_l2_error_with_yaw(scene: Dict[str, Any]) -> None:
    """
    Tests ego expert l2 error with yaw is expected value.
    :param scene: the json scene
    """
    metric = EgoExpertL2ErrorWithYawStatistics(
        'ego_expert_L2_error_with_yaw', 'Dynamics', discount_factor=1.0, heading_diff_weight=2.5
    )
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/ego_expert_l2_error_with_yaw/ego_expert_l2_error_with_yaw_zero.json')
def test_ego_expert_l2_error_with_yaw_zero(scene: Dict[str, Any]) -> None:
    """
    Tests ego expert l2 error with yaw is zero.
    :param scene: the json scene
    """
    metric = EgoExpertL2ErrorWithYawStatistics(
        'ego_expert_L2_error_with_yaw', 'Dynamics', discount_factor=1.0, heading_diff_weight=2.5
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
