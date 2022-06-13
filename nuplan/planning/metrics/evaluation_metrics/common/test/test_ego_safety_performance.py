from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_violation import DrivableAreaViolationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_min_distance_to_lead_agent import EgoMinDistanceToLeadAgent
from nuplan.planning.metrics.evaluation_metrics.common.ego_safety_performance import EgoSafetyStatistics
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision import TimeToCollisionStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/ego_safety_performance/ego_safety_performance.json')
def test_ego_safety_performance(scene: Dict[str, Any]) -> None:
    """
    Tests ego fails safety requirement (too close to lead agent)
    :param scene: the json scene
    """
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    ego_at_fault_collisions_metric = EgoAtFaultCollisionStatistics(
        'ego_at_fault_collisions_statistics', 'Dynamics', ego_lane_change_metric, max_violation_threshold=0
    )
    ego_at_fault_collisions_metric.compute(history, mock_abstract_scenario)[0]

    time_to_collision_metric = TimeToCollisionStatistics(
        'time_to_collision_statistics',
        'Planning',
        ego_lane_change_metric,
        ego_at_fault_collisions_metric,
        time_step_size=0.1,
        time_horizon=10.0,
        least_min_ttc=0.95,
    )
    time_to_collision_metric.compute(history, mock_abstract_scenario)[0]

    drivable_area_violation_metric = DrivableAreaViolationStatistics('drivable_area_violation', 'Violations', 1)
    drivable_area_violation_metric.compute(history, mock_abstract_scenario)[0]

    ego_min_distance_to_lead_agent_metric = EgoMinDistanceToLeadAgent(
        'ego_min_distance_to_lead_agent_statistics',
        'Planning',
        ego_at_fault_collisions_metric,
        min_front_distance=1.5,
        lateral_distance_threshold=4.0,
    )
    ego_min_distance_to_lead_agent_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoSafetyStatistics(
        name='ego_safety_performance_statistics',
        category='Dynamics',
        time_to_collision_metric=time_to_collision_metric,
        drivable_area_violation_metric=drivable_area_violation_metric,
        ego_at_fault_collisions_metric=ego_at_fault_collisions_metric,
        ego_min_distance_to_lead_agent_metric=ego_min_distance_to_lead_agent_metric,
    )

    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
