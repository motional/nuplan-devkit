from typing import Any, Dict, List

import pytest

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

scenario = get_test_nuplan_scenario(use_multi_sample=True)


@nuplan_test(path='json/idm_agent_observation/baseline.json')
def test_idm_observations(scene: Dict[str, Any]) -> None:
    """
    Overall integration test of IDM smart agents
    """
    # Sample
    simulation_step = 17  # fixed simulation step

    # Create Observation
    observation = IDMAgents(
        target_velocity=10,
        min_gap_to_lead_agent=0.5,
        headway_time=1.5,
        accel_max=1.0,
        decel_max=2.0,
        scenario=scenario,
        open_loop_detections_types=[],
    )

    # Simulate
    agent_history: Dict[str, List[StateSE2]] = dict()

    for step in range(simulation_step):
        iteration = SimulationIteration(time_point=scenario.get_time_point(step), index=step)
        next_iteration = SimulationIteration(time_point=scenario.get_time_point(step + 1), index=step + 1)
        buffer = SimulationHistoryBuffer.initialize_from_list(
            1,
            [scenario.get_ego_state_at_iteration(step)],
            [scenario.get_tracked_objects_at_iteration(step)],
            sample_interval=next_iteration.time_point.time_s - iteration.time_point.time_s,
        )
        observation.update_observation(iteration, next_iteration, buffer)
        for agent_id, agent in observation._idm_agent_manager.agents.items():
            if agent_id not in agent_history:
                agent_history[agent_id] = []
            agent_history[agent_id].append(agent.to_se2())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
