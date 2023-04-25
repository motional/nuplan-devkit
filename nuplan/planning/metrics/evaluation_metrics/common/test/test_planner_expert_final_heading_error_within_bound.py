from typing import Any, Dict

import pytest

from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error_within_bound import (
    PlannerExpertAverageL2ErrorStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_final_heading_error_within_bound import (
    PlannerExpertFinalHeadingErrorStatistics,
)
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/planner_expert_final_heading_error_within_bound/low_final_heading_error.json')
def test_planner_expert_final_heading_error(scene: Dict[str, Any]) -> None:
    """
    Tests planner_expert_final_heading_error is expected value.
    :param scene: the json scene.
    """
    planner_expert_average_l2_error_within_bound = PlannerExpertAverageL2ErrorStatistics(
        'planner_expert_average_l2_error',
        'Planning',
        comparison_horizon=[3, 5, 8],
        comparison_frequency=1,
        max_average_l2_error_threshold=8,
    )
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    planner_expert_average_l2_error_within_bound.compute(history, mock_abstract_scenario)
    metric = PlannerExpertFinalHeadingErrorStatistics(
        'planner_expert_final_heading_error_within_bound',
        'Planning',
        planner_expert_average_l2_error_within_bound,
        max_final_heading_error_threshold=0.8,
    )
    metric_statistic_test(scene, metric, history, mock_abstract_scenario)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
