from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_lane_change/ego_lane_change.json')
def test_ego_lane_change(scene: Dict[str, Any]) -> None:
    """
    Tests ego lane change statistics as expected.
    :param scene: the json scene
    """
    metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
