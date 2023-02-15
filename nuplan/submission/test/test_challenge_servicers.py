import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.submission.challenge_pb2 import PlannerInput as SerializedPlannerInput
from nuplan.submission.challenge_pb2 import SimulationHistoryBuffer as SerializedHistoryBuffer
from nuplan.submission.challenge_pb2 import SimulationIteration as SerializedSimulationIteration
from nuplan.submission.challenge_servicers import DetectionTracksChallengeServicer


class TestDetectionTracksChallengeServicer(TestCase):
    """Tests the DetectionTracksChallengeServicer class"""

    @patch("nuplan.submission.challenge_servicers.MapManager", return_value="map")
    def setUp(self, mock_map_manager: Mock) -> None:
        """Sets variables for testing"""
        mock_planner_cfg = {'planner1': Mock()}

        self.servicer = DetectionTracksChallengeServicer(mock_planner_cfg, mock_map_manager)

    @patch("nuplan.submission.challenge_servicers.MapManager", return_value="map")
    def test_initialization(self, mock_map_manager: Mock) -> None:
        """Tests that the class is initialized as intended."""
        mock_planner_cfg = Mock()

        mock_servicer = DetectionTracksChallengeServicer(mock_planner_cfg, mock_map_manager)

        self.assertEqual(mock_servicer.planner, None)
        self.assertEqual(mock_servicer._planner_config, mock_planner_cfg)
        self.assertEqual(mock_servicer.map_manager, mock_map_manager)

    @patch("nuplan.submission.challenge_servicers.MapManager")
    @patch("nuplan.submission.challenge_servicers.PlannerInitialization", autospec=True)
    @patch("nuplan.submission.challenge_servicers.se2_from_proto_se2")
    @patch("nuplan.submission.challenge_servicers.build_planners")
    def test_InitializePlanner(
        self, builder: Mock, mock_s2_conversion: Mock, mock_planner_initialization: Mock, mock_map_manager: Mock
    ) -> None:
        """Tests the client call to InitializePlanner."""
        mock_input = Mock()
        mock_context = Mock()
        mock_map_api = Mock()
        mock_planner_initialization.return_value = "planner_initialization"
        mock_map_manager.return_value = mock_map_api
        builder.return_value = [Mock()]

        self.servicer.InitializePlanner(mock_input, mock_context)

        calls = [
            call(mock_input.mission_goal),
        ]
        mock_s2_conversion.assert_has_calls(calls)
        map_calls = [
            call(mock_input.map_name),
            call().initialize_all_layers(),
        ]
        self.servicer.map_manager.get_map.assert_has_calls(map_calls)
        self.servicer.planner.initialize.assert_called_once_with("planner_initialization")

    def test_ComputeTrajectory_uninitialized(self) -> None:
        """Tests the client call to ComputeTrajectory fails if the planner wasn't initialized."""
        with self.assertRaises(AssertionError, msg="Planner has not been initialized. Please call InitializePlanner"):
            self.servicer.simulation_history_buffers = []
            self.servicer.ComputeTrajectory(Mock(), Mock())

    @patch("nuplan.submission.challenge_servicers.proto_traj_from_inter_traj")
    def test_ComputeTrajectory(self, proto_traj_from_inter_traj: Mock) -> None:
        """Tests the client call to ComputeTrajectory."""
        # Call setup
        mock_context = Mock()
        self.servicer.planner = Mock(spec=AbstractPlanner)
        self.servicer.planner.compute_trajectory.return_value = "trajectory"
        self.servicer.simulation_history_buffer = "buffer_1"
        self.servicer._initialized = True

        history_buffer = MagicMock(ego_states=["ego_state_1"], observations=["observation_1"])
        simulation_iteration = MagicMock(time_us=123, index=234)
        mock_serialized_input = MagicMock(
            simulation_history_buffer=history_buffer, simulation_iteration=simulation_iteration
        )

        with patch.object(self.servicer, '_build_planner_input', autospec=True) as build_planner_input:
            # Function call
            result = self.servicer.ComputeTrajectory(mock_serialized_input, mock_context)

            # Post call checks
            build_planner_input.assert_called_with(mock_serialized_input, "buffer_1")
            self.servicer.planner.compute_trajectory.assert_called_with(build_planner_input.return_value)
            proto_traj_from_inter_traj.assert_called_with(self.servicer.planner.compute_trajectory.return_value)
            self.assertEqual(proto_traj_from_inter_traj.return_value, result)

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
            result = self.servicer._build_planner_input(mock_message, buffer)

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
            result = self.servicer._build_planner_input(mock_message, None)

            # Post call checks
            buffer.initialize_from_list.assert_called_once_with(
                1, ['deserialized_ego_state'], ['deserialized_observations'], ['sample_interval']
            )

            self.assertEqual(planner_input.return_value, result)


if __name__ == '__main__':
    unittest.main()
