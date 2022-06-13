from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_relative_progress_to_goal import (
    EgoRelativeProgressToGoalStatistics,
)
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_relative_progress_to_goal/ego_relative_progress_to_goal.json')
def test_ego_relative_progress_to_goal(scene: Dict[str, Any]) -> None:
    """
    Tests ego relative progress to goal statistics as expected
    :param scene: the json scene.
    """
    metric = EgoRelativeProgressToGoalStatistics(
        'ego_relative_progress_to_goal_statistics', 'Dynamics', min_relative_progress_rate=0.5
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
