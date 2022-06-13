from typing import Any, Dict

import pytest

from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.clearance_from_static_agents import (
    ClearanceFromStaticAgentsStatistics,
)
from nuplan.planning.metrics.utils.testing_utils import metric_statistic_test


@nuplan_test(path='json/clearance_from_static_agents/clearance_from_static_agents.json')
def test_ego_expected_clearance(scene: Dict[str, Any]) -> None:
    """
    Tests clearance from static agents by checking if it is the expected clearance.
    :param scene: the json scene
    """
    metric = ClearanceFromStaticAgentsStatistics(
        'clearance_from_static_agent_statistics', 'Planning', lateral_distance_threshold=8.0
    )
    metric_statistic_test(scene=scene, metric=metric)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
