from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_distance_to_goal import EgoDistanceToGoalStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_distance_to_goal/ego_distance_to_goal.json')
def test_ego_distance_to_goal(scene: Dict[str, Any]) -> None:
    """
    Tests ego distance to goal is expected value.
    :param scene: the json scene
    """
    metric = EgoDistanceToGoalStatistics('ego_distance_to_goal_statistics', 'Planning', score_distance_threshold=5)
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
