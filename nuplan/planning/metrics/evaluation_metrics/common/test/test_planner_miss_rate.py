from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error import (
    PlannerExpertAverageL2ErrorStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.planner_miss_rate import PlannerMissRateStatistics
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/planner_miss_rate/planner_miss_rate.json')
def test_planner_miss_rate(scene: Dict[str, Any]) -> None:
    """
    Tests planner_miss_rate is expected value.
    :param scene: the json scene.
    """
    planner_expert_average_l2_error_metric = PlannerExpertAverageL2ErrorStatistics(
        'planner_expert_average_l2_error', 'Planning', comparison_horizon=4, comparison_frequency=1
    )
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    planner_expert_average_l2_error_metric.compute(history, mock_abstract_scenario)
    metric = PlannerMissRateStatistics(
        'planner_miss_rate_statistics',
        'Planning',
        planner_expert_average_l2_error_metric,
        max_displacement_threshold=2.0,
        max_miss_rate_threshold=0.3,
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
