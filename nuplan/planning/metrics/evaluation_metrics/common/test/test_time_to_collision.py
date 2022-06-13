import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision import TimeToCollisionStatistics
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/time_to_collision/time_to_collision.json')
def test_time_to_collision(scene) -> None:  # type: ignore
    """
    Test predicted time to collision
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)

    ego_at_fault_collisions_metric = EgoAtFaultCollisionStatistics(
        'ego_at_fault_collisions_statistics', 'Dynamics', ego_lane_change_metric, max_violation_threshold=0
    )

    metric = TimeToCollisionStatistics(
        'time_to_collision_statistics',
        'Planning',
        ego_lane_change_metric,
        ego_at_fault_collisions_metric,
        **scene['metric_parameters'],
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
