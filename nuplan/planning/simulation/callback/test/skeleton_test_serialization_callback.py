import lzma
import pathlib
import pickle
import tempfile
import unittest
from typing import Generator
from unittest.mock import Mock

import msgpack
import ujson as json
from hypothesis import given, settings
from hypothesis import strategies as st

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.callback.serialization_callback import SerializationCallback
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class SkeletonTestSerializationCallback(unittest.TestCase):
    """Base class for TestsSerializationCallback* classes."""

    def _setUp(self) -> None:
        """Setup mocks for our tests."""
        self._serialization_type_to_extension_map = {
            "json": ".json",
            "pickle": ".pkl.xz",
            "msgpack": ".msgpack.xz",
        }

        # Check that we're using supported serialization type.
        self._serialization_type = getattr(self, "_serialization_type", "")  # work around mypy attr-defined issue
        self.assertIn(self._serialization_type, self._serialization_type_to_extension_map)

        self.output_folder = tempfile.TemporaryDirectory()
        self.callback = SerializationCallback(
            output_directory=self.output_folder.name,
            folder_name="sim",
            serialization_type=self._serialization_type,
            serialize_into_single_file=True,
        )

        self.sim_manager = Mock(spec=AbstractSimulationTimeController)
        self.observation = Mock(spec=AbstractObservation)
        self.controller = Mock(spec=AbstractEgoController)

        super().setUp()

    @settings(deadline=None)
    @given(
        # mock_timestamp values are selected to try to trigger overflow error
        # max_value is the maximum value of 64-bit unsigned integer, the maximum value supported by msgpack.
        mock_timestamp=st.one_of(st.just(0), st.integers(min_value=1627066061949808, max_value=18446744073709551615))
    )
    def _dump_test_scenario(self, mock_timestamp: int) -> None:
        """
        Tests whether a scene can be dumped into a file and check that the keys are in the dumped scene.
        :param mock_timestamp: Mocked timestamp to pass to mock_get_traffic_light_status_at_iteration.
        """

        def mock_get_traffic_light_status_at_iteration(iteration: int) -> Generator[TrafficLightStatusData, None, None]:
            """Mocks MockAbstractScenario.get_traffic_light_status_at_iteration to return large numbers."""
            dummy_tl_data = TrafficLightStatusData(
                status=TrafficLightStatusType.GREEN,
                lane_connector_id=1,
                timestamp=mock_timestamp,
            )
            yield dummy_tl_data

        # Generate scenario & setup mocks.
        scenario = MockAbstractScenario()
        scenario.get_traffic_light_status_at_iteration = Mock(spec=scenario.get_traffic_light_status_at_iteration)
        scenario.get_traffic_light_status_at_iteration.side_effect = mock_get_traffic_light_status_at_iteration

        self.setup = SimulationSetup(
            observations=self.observation,
            scenario=scenario,
            time_controller=self.sim_manager,
            ego_controller=self.controller,
        )

        planner = Mock()
        planner.name = Mock(return_value="DummyPlanner")

        # Make sure the directory is correct
        directory = self.callback._get_scenario_folder(planner.name(), scenario)
        self.assertEqual(
            str(directory),
            self.output_folder.name + "/sim/DummyPlanner/mock_scenario_type/mock_log_name/mock_scenario_name",
        )

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
                traffic_light_status=scenario.get_traffic_light_status_at_iteration(0),
            )
        )
        history.add_sample(
            SimulationHistorySample(
                iteration=SimulationIteration(time_point=TimePoint(0), index=0),
                ego_state=state_1,
                trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
                observation=DetectionsTracks(TrackedObjects()),
                traffic_light_status=scenario.get_traffic_light_status_at_iteration(0),
            )
        )

        # Simulate simulation interation loop
        for data in history.data:
            self.callback.on_step_end(self.setup, planner, data)

        # Simulate end of simulation to trigger serialization
        self.callback.on_simulation_end(self.setup, planner, history)

        # Make sure that output file is written correctly
        filename = "mock_scenario_name" + self._serialization_type_to_extension_map[self._serialization_type]
        path = pathlib.Path(
            self.output_folder.name
            + "/sim/DummyPlanner/mock_scenario_type/mock_log_name/mock_scenario_name/"
            + filename
        )
        self.assertTrue(path.exists())

        # Make sure the important fields are in the test
        # 1. load file contents
        if self._serialization_type == "json":
            with open(path.absolute()) as f:
                data = json.load(f)
        elif self._serialization_type == "msgpack":
            with lzma.open(str(path), "rb") as f:  # type: ignore
                data = msgpack.unpackb(f.read())
        elif self._serialization_type == "pickle":
            with lzma.open(str(path), "rb") as f:  # type: ignore
                data = pickle.load(f)  # type: ignore

        # 2. make sure structure is sound
        self.assertTrue(len(data) > 0)
        data = data[0]
        self.assertTrue("world" in data.keys())
        self.assertTrue("ego" in data.keys())
        self.assertTrue("trajectories" in data.keys())
        self.assertTrue("map" in data.keys())

        # 3. make sure that we can serialize big integer values
        expected_traffic_light_data = next(scenario.get_traffic_light_status_at_iteration(0))
        actual_traffic_light_data_dict = data["traffic_light_status"][0]
        self.assertEqual(actual_traffic_light_data_dict["timestamp"], expected_traffic_light_data.timestamp)
