from typing import Any, Dict, cast

import pytest
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.metrics.evaluation_metrics.common.time_gap_to_agent import TimeGapToLeadAgent, TimeGapToRearAgent
from nuplan.planning.metrics.metric_result import MetricStatisticsType
from nuplan.planning.metrics.utils.testing_utils import setup_history
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import get_closest_agent_in_position, \
    is_agent_ahead, is_agent_behind
from pytest import approx


@nuplan_test(path='json/time_gap_to_agent/agent_front.json')
def test_time_gap_to_lead_agent(scene: Dict[str, Any]) -> None:
    """
    Test agent_relative_position
    """
    history = setup_history(scene)
    # Metric
    time_gap = TimeGapToLeadAgent("time_gap_to_lead_agent", '')
    result = time_gap.compute(history)
    closest_agent, closest_distance = get_closest_agent_in_position(
        history.data[0].ego_state, cast(Detections, history.data[0].observation), is_agent_ahead)
    assert closest_distance == approx(scene["expected"]["closest_distance"])
    assert result[0].time_series is not None
    assert result[0].statistics[MetricStatisticsType.MIN].value == approx(closest_distance /
                                                                          history.data[
                                                                              0].ego_state.dynamic_car_state.speed)


@nuplan_test(path='json/time_gap_to_agent/agent_back.json')
def test_time_gap_to_rear_agent(scene: Dict[str, Any]) -> None:
    """
    Test agent_relative_position
    """
    history = setup_history(scene)
    # Metric
    time_gap = TimeGapToRearAgent("time_gap_to_rear_agent", '')
    result = time_gap.compute(history)
    closest_agent, closest_distance = get_closest_agent_in_position(
        history.data[0].ego_state, cast(Detections, history.data[0].observation), is_agent_behind)
    assert closest_distance == approx(scene["expected"]["closest_distance"])
    assert result[0].time_series is not None
    assert result[0].statistics[MetricStatisticsType.MIN].value == \
        approx(closest_distance / max(0.1, closest_agent.velocity[0]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
