import itertools
import os
from typing import Any, Dict, List

import pytest
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_agent import IDMAgent
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_agent_manager import IDMAgentManager, \
    UniqueIDMAgents
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_agents_builder import build_map_rails
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_policy import IDMPolicy
from nuplan.planning.simulation.observation.smart_agents.occupancy_map.strtree_occupancy_map import \
    STRTreeOccupancyMapFactory
from nuplan.planning.utils.serialization.from_scene import from_scene_to_agents, to_agent_state_from_scene

map_factory = NuPlanMapFactory(
    GPKGMapsDB('nuplan-maps-v0.1', map_root=os.path.join(os.getenv('NUPLAN_DATA_ROOT', "~/nuplan/dataset"), 'maps')))

policy = IDMPolicy(target_velocity=10, min_gap_to_lead_agent=3,
                   headway_time=2, accel_max=1.0, decel_max=2.0)


def _scene_to_agents(scene: Dict[str, Any]) -> List[Box3D]:
    """
    A wrapper around from_scene_to_agents to add token to each Box3D
    :param scene: scene dictionary
    :return: List of boxes representing all agents
    """
    agents = from_scene_to_agents(scene["world"])
    agent_id = 0
    for agent in agents:
        agent.token = str(agent_id)
        agent_id += 1
    return agents  # type: ignore


def _build_idm_agents(agents: List[Box3D], map_api: AbstractMap) -> UniqueIDMAgents:
    """
    :param agents: list of agents represented by Box3D
    :param map_api: AbstractMap
    """
    unique_agents: UniqueIDMAgents = {}

    for agent in agents:
        path, progress = build_map_rails(agent, map_api, 50)
        unique_agents[str(agent.token)] = IDMAgent(start_iteration=0,
                                                   intial_state=agent,
                                                   path=path,
                                                   path_progress=progress,
                                                   policy=policy)

    return unique_agents


def _build_idm_manager(scene: Dict[str, Any]) -> IDMAgentManager:
    """
    Builds IDMAgentManager from scene
    :param scene: scene dictionary
    :return: IDMAgentManager object
    """
    map_name = scene["map"]["area"]
    map_maps_db = map_factory.build_map_from_name(map_name)
    agents = _scene_to_agents(scene)
    unique_agents = _build_idm_agents(agents, map_maps_db)
    occupancy_map = STRTreeOccupancyMapFactory().get_from_boxes(agents)
    idm_manager = IDMAgentManager(unique_agents, occupancy_map)

    return idm_manager


@nuplan_test(path='json/idm_manager/')
@pytest.mark.skip('Map us-nv-las-vegas not availble')
def test_idm_manager(scene: Dict[str, Any]) -> None:
    """
    Test idm agent manager behaviour when ego is in lane
    """
    simulation_step = 20  # fixed simulation step

    idm_manager = _build_idm_manager(scene)
    agent_history: Dict[str, List[StateSE2]] = dict()
    ego_agent = to_agent_state_from_scene(scene["ego"], get_pacifica_parameters(), to_cog=False)

    for step in range(simulation_step):
        idm_manager.propagate_agents(ego_state=ego_agent, tspan=0.5, iteration=0)
        for agent_id, agent in idm_manager.agents.items():
            if agent_id not in agent_history:
                agent_history[agent_id] = []
            agent_history[agent_id].append(agent.to_se2())

    # Check that there is no collision at the end of simulation
    for geom1, geom2 in itertools.combinations(idm_manager.agent_occupancy.get_all_geometries(), 2):
        assert not geom1.intersects(geom2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
