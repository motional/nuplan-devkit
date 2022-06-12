import pathlib
import tempfile
import unittest
from unittest.mock import Mock

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class TestSimulationLogCallback(unittest.TestCase):
    """Tests simulation_log_callback."""

    def setUp(self) -> None:
        """Setup Mocked classes."""
        self.output_folder = tempfile.TemporaryDirectory()
        self.callback = SimulationLogCallback(
            output_directory=self.output_folder.name, simulation_log_dir='simulation_log', serialization_type='msgpack'
        )

        self.sim_manager = Mock(spec=AbstractSimulationTimeController)
        self.observation = Mock(spec=AbstractObservation)
        self.controller = Mock(spec=AbstractEgoController)

    def tearDown(self) -> None:
        """Clean up folder."""
        self.output_folder.cleanup()

    def test_callback(self) -> None:
        """Tests whether a scene can be dumped into a simulation log and check the keys are correct."""
        scenario = MockAbstractScenario()

        self.setup = SimulationSetup(
            observations=self.observation,
            scenario=scenario,
            time_controller=self.sim_manager,
            ego_controller=self.controller,
        )

        planner = SimplePlanner(2, 0.5, [0, 0])

        # Make sure the directory is correct
        directory = self.callback._get_scenario_folder(planner.name(), scenario)
        self.assertEqual(str(directory), self.output_folder.name + "/simulation_log/SimplePlanner/mock_scenario_type")

        # initialize callback
        self.callback.on_initialization_start(self.setup, planner)

        # Mock two iteration steps
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
        state_0 = EgoState.build_from_rear_axle(
            StateSE2(0, 0, 0),
            vehicle_parameters=scenario.ego_vehicle_parameters,
            rear_axle_velocity_2d=StateVector2D(x=0, y=0),
            rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
            tire_steering_angle=0,
            time_point=TimePoint(0),
        )
        state_1 = EgoState.build_from_rear_axle(
            StateSE2(0, 0, 0),
            vehicle_parameters=scenario.ego_vehicle_parameters,
            rear_axle_velocity_2d=StateVector2D(x=0, y=0),
            rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
            tire_steering_angle=0,
            time_point=TimePoint(1000),
        )
        history.add_sample(
            SimulationHistorySample(
                iteration=SimulationIteration(time_point=TimePoint(0), index=0),
                ego_state=state_0,
                trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
                observation=DetectionsTracks(TrackedObjects()),
            )
        )
        history.add_sample(
            SimulationHistorySample(
                iteration=SimulationIteration(time_point=TimePoint(0), index=0),
                ego_state=state_1,
                trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
                observation=DetectionsTracks(TrackedObjects()),
            )
        )

        # Simulate simulation interation loop
        for data in history.data:
            self.callback.on_step_end(self.setup, planner, data)

        # Simulate end of simulation
        self.callback.on_simulation_end(self.setup, planner, history)

        # Compressed path
        path = pathlib.Path(
            self.output_folder.name + "/simulation_log/SimplePlanner/mock_scenario_type/mock_scenario_name.msgpack.xz"
        )

        self.assertTrue(path.exists())
        simulation_log = SimulationLog.load_data(file_path=path)
        self.assertEqual(simulation_log.file_path, path)


if __name__ == '__main__':
    unittest.main()
