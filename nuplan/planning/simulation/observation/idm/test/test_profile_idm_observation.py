import logging
import unittest

from pyinstrument import Profiler

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestProfileIDM(unittest.TestCase):
    """
    Profiling test for IDM agents.
    """

    def setUp(self) -> None:
        """
        Inherited, see super class.
        """
        self.n_repeat_trials = 1
        self.display_results = True
        self.scenario = get_test_nuplan_scenario()

    def test_profile_idm_agent_observation(self) -> None:
        """Profile IDMAgents."""
        profiler = Profiler(interval=0.0001)
        profiler.start()

        # How many times to repeat runtime test
        for _ in range(self.n_repeat_trials):
            observation = IDMAgents(
                target_velocity=10,
                min_gap_to_lead_agent=0.5,
                headway_time=1.5,
                accel_max=1.0,
                decel_max=2.0,
                scenario=self.scenario,
                open_loop_detections_types=[],
            )

            for step in range(self.scenario.get_number_of_iterations() - 1):
                iteration = SimulationIteration(time_point=self.scenario.get_time_point(step), index=step)
                next_iteration = SimulationIteration(time_point=self.scenario.get_time_point(step + 1), index=step + 1)
                buffer = SimulationHistoryBuffer.initialize_from_list(
                    1,
                    [self.scenario.get_ego_state_at_iteration(step)],
                    [self.scenario.get_tracked_objects_at_iteration(step)],
                    next_iteration.time_point.time_s - iteration.time_point.time_s,
                )
                observation.update_observation(iteration, next_iteration, buffer)
        profiler.stop()

        if self.display_results:
            logger.info(profiler.output_text(unicode=True, color=True))


if __name__ == "__main__":
    unittest.main()
