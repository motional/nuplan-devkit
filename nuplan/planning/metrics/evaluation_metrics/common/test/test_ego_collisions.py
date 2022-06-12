from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_collisions import EgoCollisionStatistics
from nuplan.planning.metrics.metric_result import MetricStatisticsType
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/ego_collision/agent_not_colliding.json')
def test_ego_not_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is no collision happens as expected.
    :param scene: the json scene
    """
    metric = EgoCollisionStatistics('ego_collision_statistics', 'Planning', max_violation_threshold=1)
    metric_result = metric_statistic_test(scene=scene, metric=metric)
    statistics = metric_result.statistics

    # No MAX when there is no collision
    assert MetricStatisticsType.MAX not in statistics


@nuplan_test(path='json/ego_collision/agent_once_colliding.json')
def test_ego_once_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is one collision in this scene.
    :param scene: the json scene
    """
    metric = EgoCollisionStatistics('ego_collision_statistics', 'Planning', max_violation_threshold=1)
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/ego_collision/agent_continuous_colliding.json')
def test_ego_continuous_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is only one collision even continuous colliding in this scene.
    :param scene: the json scene
    """
    metric = EgoCollisionStatistics('ego_collision_statistics', 'Planning', max_violation_threshold=1)
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/ego_collision/agent_colliding_twice.json')
def test_ego_twice_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is two collisions in this scene.
    :param scene: the json scene
    """
    metric = EgoCollisionStatistics('ego_collision_statistics', 'Planning', max_violation_threshold=1)
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/ego_collision/agent_colliding_all.json')
def test_ego_all_collision(scene: Dict[str, Any]) -> None:
    """
    Tests there is four collisions when combining all agents in this scene.
    :param scene: the json scene
    """
    metric = EgoCollisionStatistics('ego_collision_statistics', 'Planning', max_violation_threshold=1)
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
