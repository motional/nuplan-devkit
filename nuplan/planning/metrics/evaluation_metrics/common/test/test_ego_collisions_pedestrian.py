from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_at_fault_collisions_with_pedestrians import (
    EgoAtFaultCollisionPedestrianStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/ego_at_fault_collision/collisions_with_pedestrian.json')
def test_no_collision_with_pedestrian(scene: Dict[str, Any]) -> None:
    """
    Tests there is no collision as expected.
    :param scene: the json scene
    """
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    _ = ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    ego_at_fault_collisions_metric = EgoAtFaultCollisionStatistics(
        'ego_at_fault_collisions_statistics', 'Dynamics', ego_lane_change_metric
    )
    _ = ego_at_fault_collisions_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoAtFaultCollisionPedestrianStatistics(
        'ego_at_fault_collisions_with_pedestrians_statistics',
        'Dynamics',
        ego_at_fault_collisions_metric,
        max_violation_threshold=0,
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
