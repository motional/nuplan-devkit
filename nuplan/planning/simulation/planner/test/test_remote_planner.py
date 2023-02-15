import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.remote_planner import RemotePlanner
from nuplan.submission.challenge_pb2_grpc import DetectionTracksChallengeStub
from nuplan.submission.submission_container_manager import SubmissionContainerManager


class TestRemotePlanner(TestCase):
    """Tests RemotePlanner class"""

    @patch("nuplan.planning.simulation.planner.remote_planner.SubmissionContainerManager", autospec=True)
    def setUp(self, mock_factory: Mock) -> None:
        """Sets variables for testing"""
        self.planner = RemotePlanner()
        self.planner_with_container = RemotePlanner(
            submission_container_manager=Mock(), submission_image="foo", container_name="bar"
        )

    @patch("nuplan.planning.simulation.planner.remote_planner.SubmissionContainerManager", autospec=True)
    def test_initialization(self, mock_factory: Mock) -> None:
        """Tests that the class is initialized as intended."""
        # Test without starting container
        mock_planner = RemotePlanner()
        self.assertEqual(None, mock_planner.submission_container_manager)
        self.assertEqual(50051, mock_planner.port)

        # Test with starting container
        mock_planner = RemotePlanner(
            submission_container_manager=mock_factory, submission_image="foo", container_name="bar"
        )
        self.assertEqual(mock_factory, mock_planner.submission_container_manager)
        self.assertEqual("foo", mock_planner.submission_image)
        self.assertEqual("bar", mock_planner.container_name)
        self.assertEqual(None, mock_planner.port)

        # Test fails with missing parameters
        with self.assertRaises(AssertionError):
            _ = RemotePlanner(submission_container_manager=Mock())

    def test_name(self) -> None:
        """Tests planner name is set correctly"""
        self.assertEqual("RemotePlanner", self.planner.name())

    def test_observation_type(self) -> None:
        """Tests observation type is set correctly"""
        self.assertEqual(DetectionsTracks, self.planner.observation_type())

    def test_initialization_message_creation(self) -> None:
        """Tests that the message for the initialization request is built correctly."""
        mock_state_1 = Mock(x=0, y=1, heading=0.2)
        mock_map_api = Mock(map_name="test")

        mock_initialization = Mock(mission_goal=mock_state_1, map_api=mock_map_api, route_roadblock_ids=['a', 'b', 'c'])

        # Check raises on missing mission goal
        with self.assertRaises(AttributeError):
            self.planner._planner_initializations_to_message(Mock(mission_goal=None, map_api=mock_map_api))

        initialization_message = self.planner._planner_initializations_to_message(mock_initialization)

        self.assertAlmostEqual(mock_state_1.x, initialization_message.mission_goal.x)
        self.assertAlmostEqual(mock_state_1.y, initialization_message.mission_goal.y)
        self.assertAlmostEqual(mock_state_1.heading, initialization_message.mission_goal.heading)
        self.assertEqual(mock_map_api.map_name, initialization_message.map_name)
        self.assertEqual(initialization_message.route_roadblock_ids, ['a', 'b', 'c'])

    @patch.object(RemotePlanner, "_planner_initializations_to_message", return_value=123, autospec=True)
    @patch("grpc.insecure_channel")
    @patch("nuplan.submission.challenge_pb2_grpc.DetectionTracksChallengeStub", autospec=True)
    @patch(
        "nuplan.planning.simulation.planner.remote_planner.SubmissionContainerManager",
        Mock(spec_set=SubmissionContainerManager),
    )
    @patch("nuplan.planning.simulation.planner.remote_planner.find_free_port_number")
    def test_initialize(
        self, mock_find_port: Mock, mock_stub_function: Mock, mock_channel: Mock, initialization_to_message: Mock
    ) -> None:
        """Tests that the initialization request is called correctly."""
        mock_initialization = Mock()
        mock_stub = Mock()
        mock_stub_function.return_value = mock_stub
        self.planner.initialize(mock_initialization)

        mock_channel.assert_called()
        initialization_to_message.assert_called_with(mock_initialization)

        self.planner._stub.InitializePlanner.assert_called_with(123)

        # Test container starts when specified in parameters
        self.planner_with_container.initialize(mock_initialization)
        self.planner_with_container.submission_container_manager.get_submission_container.assert_called_with(
            self.planner_with_container.submission_image, self.planner_with_container.container_name, mock_find_port()
        )
        self.planner_with_container.submission_container_manager.get_submission_container().start.assert_called()

    @patch.object(RemotePlanner, "_compute_trajectory")
    @patch("grpc.insecure_channel", Mock())
    @patch(
        "nuplan.submission.challenge_pb2_grpc.DetectionTracksChallengeStub", Mock(spec_set=DetectionTracksChallengeStub)
    )
    def test_compute_trajectory_interface(self, mock_compute_trajectory: Mock) -> None:
        """Tests that the interface for the trajectory computation request is called correctly."""
        mock_compute_trajectory.return_value = "trajectories"
        mock_input = [Mock()]

        trajectories = self.planner.compute_trajectory(mock_input)

        mock_compute_trajectory.assert_called_with(self.planner._stub, current_input=mock_input)
        self.assertEqual("trajectories", trajectories)

    @patch("nuplan.planning.simulation.planner.remote_planner.interp_traj_from_proto_traj", Mock)
    @patch("nuplan.planning.simulation.planner.remote_planner.proto_tl_status_data_from_tl_status_data")
    @patch("nuplan.submission.challenge_pb2.PlannerInput")
    @patch("nuplan.submission.challenge_pb2.SimulationIteration")
    @patch("nuplan.submission.challenge_pb2.SimulationHistoryBuffer")
    def test_compute_trajectory(
        self,
        history_buffer: Mock,
        simulation_iteration: Mock,
        planner_input: Mock,
        mock_proto_tl_status_data: Mock,
    ) -> None:
        """Tests deserialization and serialization of the input/output for the trajectory computation interface."""
        with patch.object(self.planner, '_get_history_update', MagicMock()) as get_history_update:
            get_history_update.return_value = [["states"], ["observations"], ["intervals"]]

            mock_stub = MagicMock()
            mock_tl_data = Mock()
            mock_input = Mock(
                iteration=Mock(time_us=1, index=0),
                history=Mock(ego_states="fake_input"),
                traffic_light_data=[mock_tl_data],
            )
            mock_input.history.ego_states = ["fake_input"]

            planner_input.return_value = "planner input"

            simulation_iteration.return_value = "iter_1"
            history_buffer.return_value = "hb_1"

            self.planner._compute_trajectory(mock_stub, mock_input)

            # Checks
            get_history_update.assert_called_once_with(mock_input)
            mock_proto_tl_status_data.assert_called_once_with(mock_tl_data)
            simulation_iteration.assert_has_calls([call(time_us=1, index=0)])
            planner_input.assert_has_calls(
                [
                    call(
                        simulation_iteration="iter_1",
                        simulation_history_buffer="hb_1",
                        traffic_light_data=[mock_proto_tl_status_data.return_value],
                    )
                ]
            )
            mock_stub.ComputeTrajectory.assert_called_once_with(planner_input.return_value, timeout=1)

    @patch("pickle.dumps")
    def test_get_history_update(self, mock_dumps: Mock) -> None:
        """Tests that the history update is built correctly."""
        planner_input = Mock()
        planner_input.history.ego_states = [1, 2]
        planner_input.history.observations = [4, 5]
        planner_input.history.current_state = (6, 7)

        # Check that without cache all states are serialized
        serialized_states, serialized_observations, sample_interval = self.planner._get_history_update(planner_input)
        calls = [call(1), call(2), call(4), call(5)]
        mock_dumps.assert_has_calls(calls)

        # Check that with cache only the last states are serialized
        self.planner.serialized_state = serialized_states
        self.planner.serialized_observation = serialized_observations
        self.planner.sample_intervals = sample_interval
        _, _, _ = self.planner._get_history_update(planner_input)
        calls = calls + [call(6), call(7)]
        mock_dumps.assert_has_calls(calls)


if __name__ == '__main__':
    unittest.main()
