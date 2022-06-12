from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import (
    EgoProgressAlongExpertRouteStatistics,
)
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_progress_along_expert_route/ego_progress_along_expert_route.json')
def test_ego_progress_to_goal(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress along expert route statistics as expected.
    :param scene: the json scene
    """
    metric = EgoProgressAlongExpertRouteStatistics(
        "ego_progress_along_expert_route_statistics", "Dynamics", score_progress_threshold=10
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
