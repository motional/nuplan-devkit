import os
from typing import Any, Dict, List

import pytest
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.planning.simulation.observation.idm_agents_observation import IDMAgentsObservation
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration

db = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'))

scenario = NuPlanScenario(
    db=db,
    initial_lidar_token=db.lidar_pc[100000].token,
    subsample_ratio=None,
    scenario_extraction_info=ScenarioExtractionInfo(),
    scenario_type='unknown',
    ego_vehicle_parameters=get_pacifica_parameters())


@nuplan_test(path='json/idm_agent_observation/baseline.json')
def test_idm_observations(scene: Dict[str, Any]) -> None:
    """
    Overall integration test of IDM smart agents
    """
    # Sample
    simulation_step = 17  # fixed simulation step

    # Create Observation
    observation = IDMAgentsObservation(target_velocity=10, min_gap_to_lead_agent=0.5,
                                       headway_time=1.5, accel_max=1.0, decel_max=2.0, scenario=scenario)

    # Simulate
    agent_history: Dict[str, List[StateSE2]] = dict()

    for step in range(simulation_step):
        iteration = SimulationIteration(time_point=scenario.get_time_point(step), index=step)
        next_iteration = SimulationIteration(time_point=scenario.get_time_point(step + 1), index=step + 1)
        observation.update_observation(iteration, next_iteration, scenario.get_ego_state_at_iteration(step))
        for agent_id, agent in observation._idm_agent_manager.agents.items():
            if agent_id not in agent_history:
                agent_history[agent_id] = []
            agent_history[agent_id].append(agent.to_se2())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
