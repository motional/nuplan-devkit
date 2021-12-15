import logging
import os
import unittest

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.nuplan_db.nuplandb import NuPlanDB
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.planning.simulation.observation.idm_agents_observation import IDMAgentsObservation
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from pyinstrument import Profiler

logger = logging.getLogger(__name__)


db = NuPlanDB('nuplan_v0.1_mini', data_root=os.getenv('NUPLAN_DATA_ROOT'))

scenario = NuPlanScenario(
    db=db,
    initial_lidar_token=db.lidar_pc[100000].token,
    subsample_ratio=None,
    scenario_extraction_info=ScenarioExtractionInfo(),
    scenario_type='unknown',
    ego_vehicle_parameters=get_pacifica_parameters())


class ProfileIDM(unittest.TestCase):

    def setUp(self) -> None:
        self.n_repeat_trials = 1
        self.display_results = True

    def test_profile_idm_agent_observation(self) -> None:

        profiler = Profiler(interval=0.0001)
        profiler.start()

        # How many times to repeat runtime test
        for _ in range(self.n_repeat_trials):
            observation = IDMAgentsObservation(target_velocity=10, min_gap_to_lead_agent=0.5,
                                               headway_time=1.5, accel_max=1.0, decel_max=2.0, scenario=scenario)

            for step in range(scenario.get_number_of_iterations() - 1):
                iteration = SimulationIteration(time_point=scenario.get_time_point(step), index=step)
                next_iteration = SimulationIteration(time_point=scenario.get_time_point(step + 1), index=step + 1)
                observation.update_observation(iteration, next_iteration, scenario.get_ego_state_at_iteration(step))

        profiler.stop()

        if self.display_results:
            logger.info(profiler.output_text(unicode=True, color=True))


if __name__ == "__main__":
    unittest.main()
