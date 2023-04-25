from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_is_making_progress import EgoIsMakingProgressStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import (
    EgoProgressAlongExpertRouteStatistics,
)
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/ego_is_making_progress/ego_is_making_progress.json')
def test_ego_progress_to_goal(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress along expert route statistics as expected.
    :param scene: the json scene
    """
    ego_progress_along_expert_route_metric = EgoProgressAlongExpertRouteStatistics(
        "ego_progress_along_expert_route_statistics", "Dynamics", score_progress_threshold=0.1
    )
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    ego_progress_along_expert_route_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoIsMakingProgressStatistics(
        "ego_is_making_progress_statistics",
        "Plannning",
        ego_progress_along_expert_route_metric,
        min_progress_threshold=0.2,
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
