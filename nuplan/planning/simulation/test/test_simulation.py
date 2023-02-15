import unittest
from unittest.mock import MagicMock, call

from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)


class TestSimulation(unittest.TestCase):
    """
    Tests Simulation class which is updating simulation
    """

    def setUp(self) -> None:
        """Setup Mock classes."""
        self.scenario = MockAbstractScenario(number_of_past_iterations=10)

        self.sim_manager = StepSimulationTimeController(self.scenario)
        self.observation = TracksObservation(self.scenario)
        self.controller = PerfectTrackingController(self.scenario)

        self.setup = SimulationSetup(
            time_controller=self.sim_manager,
            observations=self.observation,
            ego_controller=self.controller,
            scenario=self.scenario,
        )
        self.simulation_history_buffer_duration = 2
        self.stepper = Simulation(
            simulation_setup=self.setup,
            callback=MultiCallback([]),
            simulation_history_buffer_duration=self.simulation_history_buffer_duration,
        )

    def test_stepper_initialize(self) -> None:
        """Test initialization method."""
        initialization = self.stepper.initialize()
        self.assertEqual(initialization.mission_goal, self.scenario.get_mission_goal())

        # Check current ego state
        self.assertEqual(
            self.stepper._history_buffer.current_state[0].rear_axle,
            self.scenario.get_ego_state_at_iteration(0).rear_axle,
        )

    def test_stepper_planner_input(self) -> None:
        """Test query to planner input function."""
        stepper = Simulation(
            simulation_setup=self.setup,
            callback=MultiCallback([]),
            simulation_history_buffer_duration=self.simulation_history_buffer_duration,
        )
        stepper.initialize()
        planner_input = stepper.get_planner_input()
        self.assertEqual(planner_input.iteration.index, 0)

    def test_run_callbacks(self) -> None:
        """Test whether all callbacks are called"""
        # Mock callback
        callback = MagicMock()

        # Create sample classes
        planner = SimplePlanner(2, 0.5, [0, 0])
        stepper = Simulation(
            simulation_setup=self.setup,
            callback=MultiCallback([callback]),
            simulation_history_buffer_duration=self.simulation_history_buffer_duration,
        )
        runner = SimulationRunner(stepper, planner)
        runner.run()

        # Make sure callback was called
        callback.on_simulation_start.assert_has_calls([call(stepper.setup)])
        callback.on_initialization_end.assert_has_calls([call(stepper.setup, planner)])
        callback.on_initialization_start.assert_has_calls([call(stepper.setup, planner)])
        callback.on_step_start.assert_has_calls([call(stepper.setup, planner)])
        callback.on_planner_start.assert_has_calls([call(stepper.setup, planner)])
        callback.on_step_end.assert_has_calls([call(stepper.setup, planner, stepper.history.last())])
        callback.on_simulation_end.assert_has_calls([call(stepper.setup, planner, stepper.history)])


if __name__ == '__main__':
    unittest.main()
