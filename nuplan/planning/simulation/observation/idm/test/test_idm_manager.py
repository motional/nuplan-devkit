import itertools
from typing import Any, Dict, List, cast

import pytest

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.observation.idm.test.utils import build_idm_manager
from nuplan.planning.utils.serialization.from_scene import to_agent_state_from_scene

maps_db = get_test_maps_db()
map_factory = NuPlanMapFactory(maps_db)
policy = IDMPolicy(target_velocity=10, min_gap_to_lead_agent=3, headway_time=2, accel_max=1.0, decel_max=2.0)


@nuplan_test(path='json/idm_manager/')
def test_idm_manager(scene: Dict[str, Any]) -> None:
    """
    Test idm agent manager behaviour when ego is in lane
    """
    simulation_step = 20  # fixed simulation step

    idm_manager = build_idm_manager(scene, map_factory, policy)
    ego_agent = to_agent_state_from_scene(scene["ego"], get_pacifica_parameters(), to_cog=False)
    traffic_light_status = {
        TrafficLightStatusType.GREEN: cast(List[str], scene["active_lane_connectors"]),
        TrafficLightStatusType.RED: cast(List[str], scene["inactive_lane_connectors"]),
    }

    for step in range(simulation_step):
        idm_manager.propagate_agents(
            ego_state=ego_agent,
            tspan=0.5,
            iteration=0,
            traffic_light_status=traffic_light_status,
            open_loop_detections=[],
            radius=100,
        )

    # Check that there is no collision at the end of simulation
    for geom1, geom2 in itertools.combinations(idm_manager.agent_occupancy.get_all_geometries(), 2):
        assert not geom1.intersects(geom2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
