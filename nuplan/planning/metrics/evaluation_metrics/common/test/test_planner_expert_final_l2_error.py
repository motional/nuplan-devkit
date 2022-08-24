from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error import (
    PlannerExpertAverageL2ErrorStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_final_l2_error import (
    PlannerExpertFinalL2ErrorStatistics,
)
from nuplan.planning.metrics.utils.testing_utils import build_mock_history_scenario_test, metric_statistic_test


@nuplan_test(path='json/planner_expert_final_l2_error/planner_expert_final_l2_error.json')
def test_planner_expert_final_l2_error(scene: Dict[str, Any]) -> None:
    """
    Tests planner_expert_final_l2_error is expected value.
    :param scene: the json scene.
    """
    planner_expert_average_l2_error_metric = PlannerExpertAverageL2ErrorStatistics(
        'planner_expert_average_l2_error',
        'Planning',
        comparison_horizon=4,
        comparison_frequency=1,
    )
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    planner_expert_average_l2_error_metric.compute(history, mock_abstract_scenario)
    metric = PlannerExpertFinalL2ErrorStatistics(
        'planner_expert_final_l2_error',
        'Planning',
        planner_expert_average_l2_error_metric,
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
