import unittest
from typing import Any
from unittest.mock import DEFAULT, Mock, PropertyMock, call, patch

from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.database.nuplan_db_orm.prediction_construction import (
    _waypoint_from_lidar_box,
    get_interpolated_waypoints,
    get_waypoints_for_agent,
    interpolate_waypoints,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class TestPredictionConstruction(unittest.TestCase):
    """Tests free function for prediction construction given future ground truth"""

    @patch("nuplan.database.nuplan_db_orm.prediction_construction.LidarBox", autospec=True)
    @patch("nuplan.database.nuplan_db_orm.prediction_construction.StateSE2", autospec=True)
    @patch("nuplan.database.nuplan_db_orm.prediction_construction.OrientedBox", autospec=True)
    @patch("nuplan.database.nuplan_db_orm.prediction_construction.StateVector2D", autospec=True)
    @patch("nuplan.database.nuplan_db_orm.prediction_construction.TimePoint", autospec=True)
    @patch("nuplan.database.nuplan_db_orm.prediction_construction.Waypoint", autospec=True)
    def test__waypoint_from_lidar_box(
        self, waypoint: Mock, time_point: Mock, state_vector: Mock, oriented_box: Mock, state_se2: Mock, lidar_box: Mock
    ) -> None:
        """Tests Waypoint creation from LidarBox"""
        # Set up mock
        lidar_box.translation = ["x", "y"]
        lidar_box.yaw = "yaw"
        lidar_box.size = ["w", "l", "h"]
        lidar_box.vx = "vx"
        lidar_box.vy = "vy"
        lidar_box.timestamp = "timestamp"

        # Function call
        result = _waypoint_from_lidar_box(lidar_box)

        # Checks
        state_se2.assert_called_once_with("x", "y", "yaw")
        oriented_box.assert_called_once_with(state_se2.return_value, width="w", length="l", height="h")
        state_vector.assert_called_once_with("vx", "vy")
        time_point.assert_called_once_with("timestamp")
        waypoint.assert_called_once_with(time_point.return_value, oriented_box.return_value, state_vector.return_value)
        self.assertEqual(waypoint.return_value, result)

    @patch("nuplan.database.nuplan_db_orm.prediction_construction.LidarBox", autospec=True)
    @patch("nuplan.database.nuplan_db_orm.prediction_construction._waypoint_from_lidar_box", autospec=True)
    def test_get_waypoints_for_agent(self, waypoint_from_lidar_box: Mock, lidar_box: Mock) -> None:
        """Tests extraction of future waypoints for a single agent"""
        end_timestamp = 5
        lidar_box.timestamp = 0

        def increase_timestamp() -> Any:
            """Increases the lidar_box timestamp"""
            lidar_box.timestamp += 1
            return DEFAULT

        type(lidar_box).next = PropertyMock(return_value=lidar_box, side_effect=increase_timestamp)

        result = get_waypoints_for_agent(lidar_box, end_timestamp)
        calls = [call(lidar_box)] * 5
        waypoint_from_lidar_box.assert_has_calls(calls)
        self.assertTrue(5, len(result))

    @patch("nuplan.database.nuplan_db_orm.prediction_construction.LidarBox", autospec=True)
    def test_get_waypoints_for_agent_empty_on_invalid_time(self, lidar_box: Mock) -> None:
        """Tests extraction of future waypoints for a single agent"""
        end_timestamp = 1
        lidar_box.timestamp = 2

        result = get_waypoints_for_agent(lidar_box, end_timestamp)

        self.assertEqual([], result)

    @patch("nuplan.database.nuplan_db_orm.prediction_construction.InterpolatedTrajectory", autospec=True)
    @patch("numpy.arange")
    @patch("nuplan.database.nuplan_db_orm.prediction_construction.TimePoint", autospec=True)
    def test_interpolate_waypoints(self, time_point: Mock, arange: Mock, interpolated_trajectory: Mock) -> None:
        """Tests interpolation of waypoints for a single agent"""
        # Setup
        waypoints = [Mock(time_us=0, spec_set=Waypoint)]
        arange.return_value = [1.12, 2.23]
        time_point.side_effect = ["tp1", "tp2"]
        trajectory_sampling = Mock(time_horizon=5, step_time=1, spec=TrajectorySampling)

        # Call tested method
        result = interpolate_waypoints(waypoints, trajectory_sampling)

        # Checks
        arange.assert_called_once_with(0, 5 * 1e6, 1 * 1e6)
        time_point_calls = [call(1), call(2)]
        time_point.assert_has_calls(time_point_calls)
        calls = [call("tp1"), call("tp2")]
        interpolated_trajectory.return_value.get_state_at_time.assert_has_calls(calls)

        self.assertEqual(result, [interpolated_trajectory.return_value.get_state_at_time.return_value] * 2)

    @patch("nuplan.database.nuplan_db_orm.prediction_construction.get_waypoints_for_agent", autospec=True)
    @patch("nuplan.database.nuplan_db_orm.prediction_construction.interpolate_waypoints", autospec=True)
    def test_get_interpolated_waypoints(
        self, mock_interpolate_waypoints: Mock, mock_get_waypoints_for_agent: Mock
    ) -> None:
        """Tests extraction and interpolation of waypoints for a list of agents"""
        box_1 = Mock(track_token='1')
        box_2 = Mock(track_token='2')
        mock_lidar_pc = Mock(timestamp=0, lidar_boxes=[box_1, box_2])
        future_trajectory_sampling = Mock(time_horizon=5)
        mock_get_waypoints_for_agent.side_effect = ["waypoints_1", "waypoints_2"]

        result = get_interpolated_waypoints(mock_lidar_pc, future_trajectory_sampling)

        get_waypoints_calls = [call(box_1, 5 * 1e6), call(box_2, 5 * 1e6)]
        mock_get_waypoints_for_agent.assert_has_calls(get_waypoints_calls)
        interpolate_waypoints_calls = [
            call("waypoints_1", future_trajectory_sampling),
            call("waypoints_2", future_trajectory_sampling),
        ]
        mock_interpolate_waypoints.assert_has_calls(interpolate_waypoints_calls)

        self.assertEqual(
            result, {'1': mock_interpolate_waypoints.return_value, '2': mock_interpolate_waypoints.return_value}
        )

    @patch("nuplan.database.nuplan_db_orm.prediction_construction.get_waypoints_for_agent", autospec=True)
    @patch("nuplan.database.nuplan_db_orm.prediction_construction.interpolate_waypoints", autospec=True)
    def test_get_interpolated_waypoints_no_waypoitns(
        self, mock_interpolate_waypoints: Mock, mock_get_waypoints_for_agent: Mock
    ) -> None:
        """Tests extraction and interpolation of waypoints for a list of agents"""
        box_1 = Mock(track_token='1')
        box_2 = Mock(track_token='2')
        mock_lidar_pc = Mock(timestamp=0, lidar_boxes=[box_1, box_2])
        future_trajectory_sampling = Mock(time_horizon=5)
        mock_get_waypoints_for_agent.side_effect = [[], ["waypoint"]]

        result = get_interpolated_waypoints(mock_lidar_pc, future_trajectory_sampling)

        get_waypoints_calls = [call(box_1, 5 * 1e6), call(box_2, 5 * 1e6)]
        mock_get_waypoints_for_agent.assert_has_calls(get_waypoints_calls)

        mock_interpolate_waypoints.assert_not_called()

        self.assertEqual(result, {'1': [], '2': []})
