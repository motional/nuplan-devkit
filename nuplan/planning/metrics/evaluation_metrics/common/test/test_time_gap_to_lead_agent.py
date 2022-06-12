from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.time_gap_to_lead_agent import TimeGapToLeadAgent
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/time_gap_to_lead_agent/agent_front.json')
def test_time_gap_to_lead_agent(scene: Dict[str, Any]) -> None:
    """
    Test time gap of ego to lead agents.
    """
    metric = TimeGapToLeadAgent('time_gap_to_lead_agent', 'Planning')
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
