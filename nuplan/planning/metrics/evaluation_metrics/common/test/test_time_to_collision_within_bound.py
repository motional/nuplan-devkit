import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import TimeToCollisionStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/time_to_collision_within_bound/time_to_collision_above_threshold.json')
def test_time_to_collision(scene) -> None:  # type: ignore
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


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
