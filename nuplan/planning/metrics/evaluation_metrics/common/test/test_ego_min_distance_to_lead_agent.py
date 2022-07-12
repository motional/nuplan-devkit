from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_min_distance_to_lead_agent import EgoMinDistanceToLeadAgent
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/ego_min_distance_to_lead_agent/ego_min_distance_to_lead_agent.json')
def test_ego_min_distance_to_lead_agent_metric(scene: Dict[str, Any]) -> None:
    """
    Tests ego fails to maintain desired min_distance to lead agent
    :param scene: the json scene
    """
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    ego_lane_change_metric = EgoLaneChangeStatistics('ego_lane_change_statistics', 'Planning', max_fail_rate=0.3)
    ego_lane_change_metric.compute(history, mock_abstract_scenario)[0]

    ego_at_fault_collisions_metric = EgoAtFaultCollisionStatistics(
        'ego_at_fault_collisions_statistics', 'Dynamics', ego_lane_change_metric
    )
    ego_at_fault_collisions_metric.compute(history, mock_abstract_scenario)[0]

    metric = EgoMinDistanceToLeadAgent(
        'ego_min_distance_to_lead_agent_statistics',
        'Planning',
        ego_at_fault_collisions_metric,
        min_front_distance=1.5,
        lateral_distance_threshold=0.3,
    )

    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
