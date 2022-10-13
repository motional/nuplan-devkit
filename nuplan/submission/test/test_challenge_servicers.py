import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

from nuplan.submission.challenge_pb2 import PlannerInput as SerializedPlannerInput
from nuplan.submission.challenge_pb2 import SimulationHistoryBuffer as SerializedHistoryBuffer
from nuplan.submission.challenge_pb2 import SimulationIteration as SerializedSimulationIteration
from nuplan.submission.challenge_servicers import DetectionTracksChallengeServicer


class TestDetectionTracksChallengeServicer(TestCase):
    """Tests the DetectionTracksChallengeServicer class"""

    @patch("nuplan.submission.challenge_servicers.MapManager", return_value="map")
    def setUp(self, mock_map_manager: Mock) -> None:
        """Sets variables for testing"""
        mock_planner = Mock(consume_batched_inputs=False)

        self.servicer = DetectionTracksChallengeServicer(mock_planner, mock_map_manager)

    @patch("nuplan.submission.challenge_servicers.MapManager", return_value="map")
    def test_initialization(self, mock_map_manager: Mock) -> None:
        """Tests that the class is initialized as intended."""
        mock_planner = Mock()

        mock_servicer = DetectionTracksChallengeServicer(mock_planner, mock_map_manager)

        self.assertEqual(mock_servicer.planner, mock_planner)
        self.assertEqual(mock_servicer.map_manager, mock_map_manager)

    @patch("nuplan.submission.challenge_servicers.MapManager")
    @patch("nuplan.submission.challenge_servicers.PlannerInitialization", autospec=True)
    @patch("nuplan.submission.challenge_servicers.se2_from_proto_se2")
    def test_InitializePlanner(
        self, mock_s2_conversion: Mock, mock_planner_initialization: Mock, mock_map_manager: Mock
    ) -> None:
        """Tests the client call to InitializePlanner."""
        initialization_1 = Mock()
        initialization_2 = Mock()
        mock_input = Mock(planner_initializations=[initialization_1, initialization_2])
        mock_context = Mock()
        mock_map_api = Mock()
        mock_planner_initialization.return_value = "planner_initialization"
        mock_map_manager.return_value = mock_map_api

        self.servicer.InitializePlanner(mock_input, mock_context)

        calls = [
            call(initialization_1.mission_goal),
            call(initialization_2.mission_goal),
        ]
        mock_s2_conversion.assert_has_calls(calls)
        map_calls = [
            call(initialization_1.map_name),
            call().initialize_all_layers(),
            call(initialization_2.map_name),
            call().initialize_all_layers(),
        ]
        self.servicer.map_manager.get_map.assert_has_calls(map_calls)
        self.servicer.planner.initialize.assert_called_once_with(["planner_initialization"] * 2)

    def test_ComputeTrajectory_uninitialized(self) -> None:
        """Tests the client call to ComputeTrajectory fails if the planner wasn't initialized."""
        with self.assertRaises(AssertionError, msg="Planner has not been initialized. Please call InitializePlanner"):
            self.servicer.simulation_history_buffers = []
            self.servicer.ComputeTrajectory(Mock(), Mock())

    @patch("nuplan.submission.challenge_servicers.proto_traj_from_inter_traj")
    @patch("nuplan.submission.challenge_servicers.chpb.MultiTrajectory")
    def test_ComputeTrajectory(self, multi_trajectory_mock: Mock, proto_traj_from_inter_traj: Mock) -> None:
        """Tests the client call to ComputeTrajectory fails if the planner wasn't initialized."""
        # Call setup
        mock_context = Mock()
        self.servicer.planner.compute_trajectory.return_value = ["trajectory"]
        self.servicer.simulation_history_buffers = ["buffer_1"]

        history_buffer = MagicMock(ego_states=["ego_state_1"], observations=["observation_1"])
        simulation_iteration = MagicMock(time_us=123, index=234)
        mock_serialized_input = MagicMock(
            simulation_history_buffer=history_buffer, simulation_iteration=simulation_iteration
        )

        with patch.object(self.servicer, '_build_planner_inputs', autospec=True) as build_planner_inputs:
            # Function call
            result = self.servicer.ComputeTrajectory(mock_serialized_input, mock_context)

            # Post call checks
            build_planner_inputs.assert_called_with(mock_serialized_input.planner_inputs)
            self.servicer.planner.compute_trajectory.assert_called_with(build_planner_inputs.return_value)
            proto_traj_from_inter_traj.assert_called_with(self.servicer.planner.compute_trajectory.return_value[0])
            multi_trajectory_mock.assert_called_once_with(trajectories=[proto_traj_from_inter_traj.return_value])
            self.assertEqual(multi_trajectory_mock.return_value, result)

    @patch("nuplan.submission.challenge_servicers.SimulationIteration", autospec=True)
    @patch("nuplan.submission.challenge_servicers.TimePoint", autospec=True)
    def test__extract_simulation_iteration(self, time_point: Mock, simulation_iteration: Mock) -> None:
        """Tests extraction of simulation iteration data from serialized message"""
        mock_iteration = Mock(time_us=123, index=456, spec_set=SerializedSimulationIteration)
        mock_message = Mock(simulation_iteration=mock_iteration, spec_set=SerializedPlannerInput)

        result = self.servicer._extract_simulation_iteration(mock_message)

        time_point.assert_called_once_with(123)
        simulation_iteration.assert_called_once_with(time_point.return_value, 456)
        self.assertEqual(simulation_iteration.return_value, result)

    @patch("pickle.loads")
    @patch("nuplan.submission.challenge_servicers.PlannerInput", autospec=True)
    @patch("nuplan.submission.challenge_servicers.SimulationHistoryBuffer", autospec=True)
    def test__build_planner_input(self, buffer: MagicMock, planner_input: Mock, loads: Mock) -> None:
        """Tests that planner input is correctly deserialized"""
        mock_serialized_buffer = Mock(
            ego_states=["ego_state"],
            observations=["observations"],
            sample_interval=["sample_interval"],
            spec_set=SerializedHistoryBuffer,
        )
        mock_message = MagicMock(simulation_history_buffer=mock_serialized_buffer, spec_set=SerializedPlannerInput)
        loads.side_effect = ["deserialized_ego_state", "deserialized_observations"]

        with patch.object(self.servicer, '_extract_simulation_iteration', autospec=True) as extract_iteration:
            # Function call
            result = self.servicer._build_planner_input(mock_message, buffer, 0)

            # Post call checks
            extract_iteration.assert_called_with(mock_message)
            loads.assert_has_calls([call("ego_state"), call("observations")])
            buffer.extend.assert_called_once_with(["deserialized_ego_state"], ["deserialized_observations"])

            self.assertEqual(planner_input.return_value, result)

    @patch("pickle.loads")
    @patch("nuplan.submission.challenge_servicers.PlannerInput", autospec=True)
    @patch("nuplan.submission.challenge_servicers.SimulationHistoryBuffer", autospec=True)
    def test__build_planner_input_no_buffer(self, buffer: MagicMock, planner_input: Mock, loads: Mock) -> None:
        """Tests that planner input is correctly deserialized"""
        mock_serialized_buffer = Mock(
            ego_states=["ego_state"],
            observations=["observations"],
            sample_interval=["sample_interval"],
            spec_set=SerializedHistoryBuffer,
        )
        mock_message = MagicMock(simulation_history_buffer=mock_serialized_buffer, spec_set=SerializedPlannerInput)
        loads.side_effect = ["deserialized_ego_state", "deserialized_observations"]

        with patch.object(self.servicer, '_extract_simulation_iteration', autospec=True):
            # Function call
            self.servicer.simulation_history_buffers = [mock_serialized_buffer]
            result = self.servicer._build_planner_input(mock_message, None, 0)

            # Post call checks
            buffer.initialize_from_list.assert_called_once_with(
                1, ['deserialized_ego_state'], ['deserialized_observations'], ['sample_interval']
            )

            self.assertEqual(planner_input.return_value, result)

    def test__build_planner_inputs(self) -> None:
        """Tests that planner inputs are correctly built in batch"""
        planner_inputs = [1, 2]
        self.servicer.simulation_history_buffers = ["buffer_1", "buffer_2"]
        calls = [call(1, "buffer_1", 0), call(2, "buffer_2", 1)]
        with patch.object(self.servicer, '_build_planner_input', autospec=True) as build_planner_input:
            build_planner_input.side_effect = ["planner_input_1", "planner_input_2"]

            result = self.servicer._build_planner_inputs(planner_inputs)
            build_planner_input.assert_has_calls(calls)

            self.assertEqual(["planner_input_1", "planner_input_2"], result)


if __name__ == '__main__':
    unittest.main()
