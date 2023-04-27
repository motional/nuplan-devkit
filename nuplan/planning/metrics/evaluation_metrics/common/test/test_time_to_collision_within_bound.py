from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import TimeToCollisionStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


def _run_time_to_collision_test(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision
    :param scene: the json scene
    """
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    no_ego_at_fault_collisions_metric = EgoAtFaultCollisionStatistics(
        'no_ego_at_fault_collisions_statistics', 'Dynamics', ego_lane_change_metric
    )
    no_ego_at_fault_collisions_metric.compute(history, mock_abstract_scenario)[0]

    metric = TimeToCollisionStatistics(
        'time_to_collision_statistics',
        'Planning',
        ego_lane_change_metric,
        no_ego_at_fault_collisions_metric,
        **scene['metric_parameters'],
    )
    metric_statistic_test(scene=scene, metric=metric)


@nuplan_test(path='json/time_to_collision_within_bound/time_to_collision_above_threshold.json')
def test_time_to_collision_above_threshold(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision when above threshold.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)


@nuplan_test(path='json/time_to_collision_within_bound/in_collision.json')
def test_time_to_collision_in_collision(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision in case where there is a collision.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)


@nuplan_test(path='json/time_to_collision_within_bound/ego_stopped.json')
def test_time_to_collision_ego_stopped(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision when ego is stopped.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)


@nuplan_test(path='json/time_to_collision_within_bound/no_collisions.json')
def test_time_to_collision_no_collisions(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision when there are relevant tracks, but ego will not collide.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)


@nuplan_test(path='json/time_to_collision_within_bound/no_relevant_tracks.json')
def test_time_to_collision_no_relevant_tracks(scene: Dict[str, Any]) -> None:
    """
    Test predicted time to collision when no relevant tracks.
    :param scene: the json scene
    """
    _run_time_to_collision_test(scene)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
