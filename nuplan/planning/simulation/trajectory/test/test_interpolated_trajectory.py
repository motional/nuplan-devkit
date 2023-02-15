import unittest
from abc import ABC
from unittest.mock import MagicMock, Mock, patch

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.utils.helpers import static_vars
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.utils.split_state import SplitState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class MockPoint(InterpolatableState, ABC):
    """Mock point for trajectory tests."""

    @staticmethod
    @static_vars(calls=[])
    def from_split_state(split_state: Mock) -> str:
        """Mock from_split_state."""
        MockPoint.from_split_state.calls.append(split_state)
        return "foo"

    @staticmethod
    def reset_calls() -> None:
        """Resets spy."""
        MockPoint.from_split_state.calls = []


class TestInterpolatedTrajectory(unittest.TestCase):
    """Tests implementation of InterpolatedTrajectory."""

    def setUp(self) -> None:
        """Test setup."""
        self.split_state_1 = Mock(linear_states=[123], angular_states=[2.13], fixed_states=["fix"], autspec=SplitState)
        self.split_state_2 = Mock(linear_states=[456], angular_states=[3.13], fixed_states=["fix"], autspec=SplitState)
        self.start_time_point = TimePoint(0)
        self.end_time_point = TimePoint(int(1e6))
        self.points = [
            MagicMock(
                time_point=self.start_time_point,
                time_us=self.start_time_point.time_us,
                to_split_state=lambda: self.split_state_1,
                spec=MockPoint,
            ),
            MagicMock(
                time_point=self.end_time_point,
                time_us=self.end_time_point.time_us,
                to_split_state=lambda: self.split_state_2,
                spec=MockPoint,
            ),
        ]
        # Set points
        self.trajectory = InterpolatedTrajectory(self.points)

    def tearDown(self) -> None:
        """Resets mock objects."""
        MockPoint.reset_calls()

    @patch("nuplan.planning.simulation.trajectory.interpolated_trajectory.sp_interp")
    @patch("nuplan.planning.simulation.trajectory.interpolated_trajectory.np")
    @patch("nuplan.planning.simulation.trajectory.interpolated_trajectory.AngularInterpolator", autospec=True)
    def test_initialization(self, mock_interp_angular: Mock, mock_np: Mock, mock_sp_interp: Mock) -> None:
        """Tests that initialization works as intended."""
        mock_sp_interp.interp1d.return_value = "interp_function"
        mock_np.array.return_value = "array"

        # Function call
        trajectory = InterpolatedTrajectory(self.points)

        self.assertEqual(trajectory._trajectory_class, MockPoint)
        self.assertEqual(trajectory._fixed_state, ["fix"])
        mock_sp_interp.interp1d.assert_called_with([0, 1000000], mock_np.array.return_value, axis=0)
        self.assertEqual(trajectory._function_interp_linear, mock_sp_interp.interp1d.return_value)
        mock_interp_angular.assert_called_with([0, 1000000], "array")
        self.assertEqual(trajectory._angular_interpolator, mock_interp_angular.return_value)

        # Check assertion error with not enough waypoints
        with self.assertRaises(AssertionError):
            InterpolatedTrajectory([MagicMock()])

    def test_start_end_time(self) -> None:
        """Tests that properties return correct members."""
        self.assertEqual(self.start_time_point, self.trajectory.start_time)
        self.assertEqual(self.end_time_point, self.trajectory.end_time)

    def test_get_state_at_time(self) -> None:
        """Tests interpolation method."""
        time_point = TimePoint(int(0.5 * 1e6))
        state = self.trajectory.get_state_at_time(time_point)
        self.assertEqual("foo", state)

        interpolated_state = SplitState(linear_states=[289.5], angular_states=[2.63], fixed_states=["fix"])
        self.assertEqual(MockPoint.from_split_state.calls, [interpolated_state])

        time_point_outside_interval = TimePoint(int(5 * 1e6))
        with self.assertRaises(AssertionError):
            self.trajectory.get_state_at_time(time_point_outside_interval)

    def test_get_state_at_times(self) -> None:
        """Tests batch interpolation method."""
        time_points = [TimePoint(0), TimePoint(int(0.5 * 1e6))]
        states = self.trajectory.get_state_at_times(time_points)
        self.assertEqual(["foo", "foo"], states)

        initial_state = SplitState(linear_states=[123], angular_states=[2.13], fixed_states=["fix"])
        interpolated_state = SplitState(linear_states=[289.5], angular_states=[2.63], fixed_states=["fix"])
        self.assertEqual(MockPoint.from_split_state.calls, [initial_state, interpolated_state])

        time_point_outside_interval = TimePoint(int(5 * 1e6))
        with self.assertRaises(AssertionError):
            self.trajectory.get_state_at_times([time_point_outside_interval])

    def test_get_sampled_trajectory(self) -> None:
        """Tests getter for entire trajectory."""
        self.assertEqual(self.points, self.trajectory.get_sampled_trajectory())


if __name__ == '__main__':
    unittest.main()
