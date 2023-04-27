from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error_within_bound import (
    PlannerExpertAverageL2ErrorStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.planner_miss_rate_within_bound import PlannerMissRateStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/planner_miss_rate_within_bound/high_miss_rate.json')
def test_planner_miss_rate(scene: Dict[str, Any]) -> None:
    """
    Tests planner_miss_rate is expected value.
    :param scene: the json scene.
    """
    planner_expert_average_l2_error_within_bound_metric = PlannerExpertAverageL2ErrorStatistics(
        'planner_expert_average_l2_error_within_bound',
        'Planning',
        comparison_horizon=[3, 5, 8],
        comparison_frequency=1,
        max_average_l2_error_threshold=8,
    )
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    planner_expert_average_l2_error_within_bound_metric.compute(history, mock_abstract_scenario)
    metric = PlannerMissRateStatistics(
        'planner_miss_rate_within_bound_statistics',
        'Planning',
        planner_expert_average_l2_error_within_bound_metric,
        max_displacement_threshold=[6, 8, 16],
        max_miss_rate_threshold=0.3,
    )
    metric_statistic_test(scene, metric, history, mock_abstract_scenario)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
