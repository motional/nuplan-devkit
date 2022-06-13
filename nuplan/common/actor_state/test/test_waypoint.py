import unittest
from unittest.mock import Mock, patch

from nuplan.common.actor_state.waypoint import Waypoint


class TestWaypoint(unittest.TestCase):
    """Tests Waypoint class"""

    def setUp(self) -> None:
        """Sets sample parameters for testing"""
        mock_time_point = Mock(time_us=0)
        mock_box = Mock(
            center=Mock(x="center_x", y="center_y", heading="center_heading"),
            length="length",
            width="width",
            height="height",
        )
        mock_velocity = Mock(x="velocity_x", y="velocity_y")

        # Providing velocity
        self.waypoint = Waypoint(mock_time_point, mock_box, mock_velocity)

        # Not providing velocity
        self.waypoint_no_vel = Waypoint(mock_time_point, mock_box)

    def test_iterable(self) -> None:
        """Test that the iterable gets built correctly."""
        iterable_waypoint = iter(self.waypoint)

        iterable_expected = [0, "center_x", "center_y", "center_heading", "velocity_x", "velocity_y"]

        for expected, actual in zip(iterable_expected, iterable_waypoint):
            self.assertEqual(expected, actual)

        iterable_waypoint_no_vel = iter(self.waypoint_no_vel)

        iterable_expected = [0, "center_x", "center_y", "center_heading", None, None]

        for expected, actual in zip(iterable_expected, iterable_waypoint_no_vel):
            self.assertEqual(expected, actual)

    def test_serialize(self) -> None:
        """Tests that the serialization works as expected."""
        serialized_waypoint = self.waypoint.serialize()
        serialized_expected = [
            0,
            "center_x",
            "center_y",
            "center_heading",
            "length",
            "width",
            "height",
            "velocity_x",
            "velocity_y",
        ]
        self.assertEqual(serialized_expected, serialized_waypoint)

        serialized_waypoint_no_vel = self.waypoint_no_vel.serialize()
        serialized_no_vel_expected = [
            0,
            "center_x",
            "center_y",
            "center_heading",
            "length",
            "width",
            "height",
            None,
            None,
        ]
        self.assertEqual(serialized_no_vel_expected, serialized_waypoint_no_vel)

    @patch("nuplan.common.actor_state.waypoint.StateVector2D")
    @patch("nuplan.common.actor_state.waypoint.OrientedBox")
    @patch("nuplan.common.actor_state.waypoint.TimePoint")
    @patch("nuplan.common.actor_state.waypoint.StateSE2")
    @patch("nuplan.common.actor_state.waypoint.Waypoint")
    def test_deserialize(
        self, mock_waypoint: Mock, mock_se2: Mock, mock_time_point: Mock, mock_box: Mock, mock_velocity: Mock
    ) -> None:
        """Tests that the object is deserialized correctly."""
        mock_se2.return_value = "se2"
        mock_time_point.return_value = "time_point"
        mock_box.return_value = "mock_box"
        mock_velocity.return_value = "velocity"

        waypoint = self.waypoint.deserialize([0, 1, 2, 3, 4, 5, 6, 7, 8])

        mock_time_point.assert_called_once_with(0)
        mock_se2.assert_called_once_with(1, 2, 3)
        mock_box.assert_called_once_with(mock_se2.return_value, 4, 5, 6)
        mock_velocity.assert_called_once_with(7, 8)

        mock_waypoint.assert_called_with(
            time_point=mock_time_point.return_value,
            oriented_box=mock_box.return_value,
            velocity=mock_velocity.return_value,
        )
        self.assertEqual(mock_waypoint.return_value, waypoint)

    @patch("nuplan.common.actor_state.waypoint.StateVector2D")
    @patch("nuplan.common.actor_state.waypoint.OrientedBox")
    @patch("nuplan.common.actor_state.waypoint.TimePoint")
    @patch("nuplan.common.actor_state.waypoint.StateSE2")
    @patch("nuplan.common.actor_state.waypoint.Waypoint")
    def test_deserialize_no_velocity(
        self, mock_waypoint: Mock, mock_se2: Mock, mock_time_point: Mock, mock_box: Mock, mock_velocity: Mock
    ) -> None:
        """Tests that the object is deserialized correctly when no velocity is provided."""
        mock_se2.return_value = "se2"
        mock_time_point.return_value = "time_point"
        mock_box.return_value = "mock_box"
        mock_velocity.return_value = "velocity"

        waypoint = self.waypoint.deserialize([0, 1, 2, 3, 4, 5, 6, None, None])

        mock_time_point.assert_called_once_with(0)
        mock_se2.assert_called_once_with(1, 2, 3)
        mock_box.assert_called_once_with(mock_se2.return_value, 4, 5, 6)
        mock_velocity.assert_not_called()

        mock_waypoint.assert_called_with(
            time_point=mock_time_point.return_value, oriented_box=mock_box.return_value, velocity=None
        )
        self.assertEqual(mock_waypoint.return_value, waypoint)

    @patch("nuplan.common.actor_state.waypoint.SplitState", autospec=True)
    def test_to_split_state(self, mock_split_state: Mock) -> None:
        """Tests that the object is split correctly"""
        result = self.waypoint.to_split_state()

        expected_linear_states = [0, "center_x", "center_y", "velocity_x", "velocity_y"]
        expected_angular_states = ["center_heading"]
        expected_fixed_states = ["width", "length", "height"]

        mock_split_state.assert_called_once_with(expected_linear_states, expected_angular_states, expected_fixed_states)

        self.assertEqual(result, mock_split_state.return_value)

    @patch("nuplan.common.actor_state.waypoint.StateVector2D", autospec=True)
    @patch("nuplan.common.actor_state.waypoint.OrientedBox", autospec=True)
    @patch("nuplan.common.actor_state.waypoint.TimePoint", autospec=True)
    @patch("nuplan.common.actor_state.waypoint.StateSE2", autospec=True)
    def test_from_split_state(self, mock_se2: Mock, mock_time_point: Mock, mock_box: Mock, mock_vector: Mock) -> None:
        """Tests that the object is recreated correctly from a split state"""
        split_state = self.waypoint.to_split_state()
        result = self.waypoint.from_split_state(split_state)

        mock_time_point.assert_called_once_with(0)
        mock_se2.assert_called_once_with("center_x", "center_y", "center_heading")
        mock_vector.assert_called_once_with("velocity_x", "velocity_y")
        mock_box.assert_called_once_with(mock_se2.return_value, length="length", width="width", height="height")

        self.assertEqual(result.time_point, mock_time_point.return_value)
        self.assertEqual(result.oriented_box, mock_box.return_value)
        self.assertEqual(result.velocity, mock_vector.return_value)


if __name__ == '__main__':
    unittest.main()
