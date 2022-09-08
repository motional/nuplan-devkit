import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import numpy.testing as np_test
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class TestLQRTracker(unittest.TestCase):
    """
    Tests LQR Tracker.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.initial_time_point = TimePoint(0)
        self.scenario = MockAbstractScenario(initial_time_us=self.initial_time_point)
        self.trajectory = InterpolatedTrajectory(list(self.scenario.get_expert_ego_trajectory()))
        self.sampling_time = 0.5
        self.tracker = LQRTracker(
            q_diag=np.ones((1, 6)).flatten(),
            r_diag=np.ones((1, 2)).flatten(),
            proportional_gain=1.0,
            look_ahead_seconds=0.5,
            look_ahead_meters=0.5,
            stopping_velocity=0.5,
        )
        self.tracker.initialize()

    def test_wrong_q_diag_dimension(self) -> None:
        """Expect raise AssertionError: q_diag has to have length of 5"""
        with self.assertRaises(AssertionError):
            LQRTracker(
                q_diag=np.ones((1, 4)).flatten(),
                r_diag=np.ones((1, 2)).flatten(),
                proportional_gain=1.0,
                look_ahead_seconds=0.5,
                look_ahead_meters=1.0,
                stopping_velocity=0.5,
            )

    def test_wrong_r_diag_dimension(self) -> None:
        """Expect raise AssertionError: r_diag has to have length of 2"""
        with self.assertRaises(AssertionError):
            LQRTracker(
                q_diag=np.ones((1, 6)).flatten(),
                r_diag=np.ones((1, 1)).flatten(),
                proportional_gain=1.0,
                look_ahead_seconds=0.5,
                look_ahead_meters=1.0,
                stopping_velocity=0.5,
            )

    def test_wrong_proportional_gain(self) -> None:
        """Expect raise AssertionError: proportional_gain has to be greater than 0"""
        with self.assertRaises(AssertionError):
            LQRTracker(
                q_diag=np.ones((1, 5)).flatten(),
                r_diag=np.ones((1, 2)).flatten(),
                proportional_gain=0.0,
                look_ahead_seconds=1.0,
                look_ahead_meters=1.0,
                stopping_velocity=0.5,
            )

    def test_wrong_look_ahead_seconds(self) -> None:
        """Expect raise AssertionError: look_ahead_seconds has to be greater than 0"""
        with self.assertRaises(AssertionError):
            LQRTracker(
                q_diag=np.ones((1, 5)).flatten(),
                r_diag=np.ones((1, 2)).flatten(),
                proportional_gain=1.0,
                look_ahead_seconds=-1.0,
                look_ahead_meters=1.0,
                stopping_velocity=0.5,
            )

    def test_wrong_look_ahead_meters(self) -> None:
        """Expect raise AssertionError: look_ahead_meters has to be greater than 0"""
        with self.assertRaises(AssertionError):
            LQRTracker(
                q_diag=np.ones((1, 5)).flatten(),
                r_diag=np.ones((1, 2)).flatten(),
                proportional_gain=1.0,
                look_ahead_seconds=1.0,
                look_ahead_meters=-1.0,
                stopping_velocity=0.5,
            )

    def test_wrong_stopping_velocity(self) -> None:
        """Expect raise AssertionError: stopping_velocity has to be greater than 0"""
        with self.assertRaises(AssertionError):
            LQRTracker(
                q_diag=np.ones((1, 5)).flatten(),
                r_diag=np.ones((1, 2)).flatten(),
                proportional_gain=1.0,
                look_ahead_seconds=0.5,
                look_ahead_meters=1.0,
                stopping_velocity=-0.5,
            )

    @patch("nuplan.planning.simulation.controller.tracker.lqr.LQRTracker._compute_control_action")
    def test_track_trajectory(self, compute_control_action: MagicMock) -> None:
        """Mock test the LQR track_trajectory function."""
        mock_accel_cmd = 1.0
        mock_steering_rate_cmd = 1.0
        compute_control_action.return_value = [mock_accel_cmd, mock_steering_rate_cmd]

        dynamic_state = self.tracker.track_trajectory(
            current_iteration=SimulationIteration(self.initial_time_point, 0),
            next_iteration=SimulationIteration(TimePoint(int(self.sampling_time * 1e6)), 1),
            initial_state=self.scenario.initial_ego_state,
            trajectory=self.trajectory,
        )

        self.assertAlmostEqual(dynamic_state.acceleration, mock_accel_cmd)
        self.assertAlmostEqual(dynamic_state.tire_steering_rate, mock_steering_rate_cmd)

    @patch("nuplan.planning.simulation.controller.tracker.lqr.LQRTracker._compute_error")
    @patch("nuplan.planning.simulation.controller.tracker.lqr.LQRTracker._compute_lqr_control_action")
    def test__compute_control_action(self, mock_compute_lqr_action: MagicMock, mock_compute_error: MagicMock) -> None:
        """Mock test the LQR _compute_control_action function."""
        mock_ego_state = Mock()
        state: npt.NDArray[np.float64] = np.array([1, 1, 1, 1, 1])
        reference: npt.NDArray[np.float64] = np.array([2, 2, 2, 2, 2])
        error: npt.NDArray[np.float64] = np.array([-1, -1, -1, -1, -1, -1])

        mock_compute_error.return_value = error

        self.tracker._compute_control_action(state, reference, mock_ego_state, self.sampling_time)
        mock_compute_error.assert_called_with(state, reference, self.sampling_time)
        mock_compute_lqr_action.assert_called_with(error, mock_ego_state, self.sampling_time)

    @patch("nuplan.planning.simulation.controller.tracker.lqr.StateSpace", autospec=True)
    @patch("nuplan.planning.simulation.controller.tracker.lqr.LQRTracker._linearize_model")
    @patch("nuplan.planning.simulation.controller.tracker.lqr.dlqr")
    def test__compute_lqr_control_action(
        self, mock_dlqr: MagicMock, mock_linearize_model: MagicMock, mock_ss: MagicMock
    ) -> None:
        """Mock test the LQR _compute_lqr_control_action function."""
        mock_ego_state = Mock()
        mock_ego_state.rear_axle.heading = 1.0
        mock_ego_state.dynamic_car_state.rear_axle_velocity_2d.x = 1.0
        mock_ego_state.tire_steering_angle = 1.0

        mock_linearize_model.return_value = (np.eye(6), np.ones((6, 2)))

        mock_gain_matrix: npt.NDArray[np.float64] = np.array([[3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3]])
        mock_dlqr.return_value = (mock_gain_matrix, None, None)
        error: npt.NDArray[np.float64] = np.array([-1, -1, -1, -1, -1, -1])
        results = -mock_gain_matrix.dot(error)

        control_action = self.tracker._compute_lqr_control_action(error, mock_ego_state, self.sampling_time)

        mock_ss.assert_called_once()
        mock_linearize_model.assert_called_with(
            heading=mock_ego_state.rear_axle.heading,
            velocity=mock_ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            steering_angle=mock_ego_state.tire_steering_angle,
        )
        mock_dlqr.assert_called_once()
        self.assertAlmostEqual(len(control_action), 2)
        self.assertAlmostEqual(control_action[0], results[0])
        self.assertAlmostEqual(control_action[1], results[1])

    def test__compute_error(self) -> None:
        """Test _compute_error function."""
        state: npt.NDArray[np.float64] = np.array([1, 1, 1, 1, 1])
        reference: npt.NDArray[np.float64] = np.array([2, 2, 2, 2, 2])
        expected: npt.NDArray[np.float64] = state - reference
        expected = np.concatenate((expected, [-self.sampling_time]))

        error = self.tracker._compute_error(state, reference, self.sampling_time)
        np_test.assert_allclose(error, expected)

        state = np.array([0, 0, np.pi / 3, 0, np.pi / 3])
        reference = np.array([0, 0, -np.pi / 3, 0, -np.pi / 3])
        expected = np.array([0, 0, 2 * np.pi / 3, 0, 2 * np.pi / 3, 0])

        error = self.tracker._compute_error(state, reference, 0)
        np_test.assert_allclose(error, expected)

        state = np.array([0, 0, 2 * np.pi / 3, 0, 2 * np.pi / 3])
        reference = np.array([0, 0, -2 * np.pi / 3, 0, -2 * np.pi / 3])
        expected = np.array([0, 0, -2 * np.pi / 3, 0, -2 * np.pi / 3, 0])

        error = self.tracker._compute_error(state, reference, self.sampling_time)
        np_test.assert_allclose(error, expected)

    def test__infer_refernce_velocity(self) -> None:
        """Test _infer_refernce_velocity"""
        result = self.tracker._infer_refernce_velocity(self.trajectory, TimePoint(int(1e6)))
        self.assertAlmostEqual(1.0, result)


if __name__ == "__main__":
    unittest.main()
