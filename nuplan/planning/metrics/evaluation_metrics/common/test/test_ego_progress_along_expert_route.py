from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
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
        "ego_progress_along_expert_route_statistics", "Dynamics", score_progress_threshold=2
    )
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/ego_progress_along_expert_route/ego_no_progress_along_expert_route.json')
def test_ego_no_progress_to_goal(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress along expert route statistics when expert isn't assigned a route at first and ego isn't making enough progress.
    :param scene: the json scene
    """
    metric = EgoProgressAlongExpertRouteStatistics(
        "ego_progress_along_expert_route_statistics", "Dynamics", score_progress_threshold=2
    )
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/ego_progress_along_expert_route/ego_no_route.json')
def test_no_route(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when there's no route.
    :param scene: the json scene
    """
    metric = EgoProgressAlongExpertRouteStatistics(
        "ego_progress_along_expert_route_statistics", "Dynamics", score_progress_threshold=2
    )
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/ego_progress_along_expert_route/ego_drives_backward.json')
def test_ego_backward_driving(scene: Dict[str, Any]) -> None:
    """
    Tests ego progress metric when ego drives backward.
    :param scene: the json scene
    """
    metric = EgoProgressAlongExpertRouteStatistics(
        "ego_progress_along_expert_route_statistics", "Dynamics", score_progress_threshold=2
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
