import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState, get_acceleration_shifted, get_velocity_shifted
from nuplan.common.actor_state.state_representation import StateVector2D


class TestDynamicCarState(unittest.TestCase):
    """Tests DynamicCarState class and helper functions"""

    def setUp(self) -> None:
        """Sets sample variables for testing"""
        self.displacement = StateVector2D(2.0, 2.0)
        self.reference_vector = StateVector2D(2.3, 3.4)  # Can be used for both velocity and acceleration
        self.angular_velocity = 0.2

        self.dynamic_car_state = DynamicCarState(
            rear_axle_to_center_dist=1,
            rear_axle_velocity_2d=self.reference_vector,
            rear_axle_acceleration_2d=StateVector2D(0.1, 0.2),
            angular_velocity=2,
            angular_acceleration=2.5,
            tire_steering_rate=0.5,
        )

    def test_velocity_transfer(self) -> None:
        """Tests behavior of velocity transfer formula for planar rigid bodies."""
        # Nominal case
        actual_velocity = get_velocity_shifted(self.displacement, self.reference_vector, self.angular_velocity)
        expected_velocity_p2 = StateVector2D(1.9, 3.8)
        np.testing.assert_array_almost_equal(expected_velocity_p2.array, actual_velocity.array, 6)

        # No displacement
        actual_velocity = get_velocity_shifted(StateVector2D(0.0, 0.0), self.reference_vector, self.angular_velocity)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_velocity.array, 6)

        # No rotation
        actual_velocity = get_velocity_shifted(self.displacement, self.reference_vector, 0)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_velocity.array, 6)

    def test_acceleration_transfer(self) -> None:
        """Tests behavior of acceleration transfer formula for planar rigid bodies."""
        # Nominal case
        angular_acceleration = 0.234
        actual_acceleration = get_acceleration_shifted(
            self.displacement, self.reference_vector, self.angular_velocity, angular_acceleration
        )
        np.testing.assert_array_almost_equal(StateVector2D(2.848, 3.948).array, actual_acceleration.array, 6)

        # No displacement
        actual_acceleration = get_acceleration_shifted(
            StateVector2D(0.0, 0.0), self.reference_vector, self.angular_velocity, angular_acceleration
        )
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_acceleration.array, 6)

        # No rotation and acceleration
        actual_acceleration = get_acceleration_shifted(self.displacement, self.reference_vector, 0, 0)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_acceleration.array, 6)

    def test_initialization(self) -> None:
        """Tests that object initialization works as intended"""
        self.assertEqual(1, self.dynamic_car_state._rear_axle_to_center_dist)
        self.assertEqual(self.reference_vector, self.dynamic_car_state._rear_axle_velocity_2d)
        self.assertEqual(StateVector2D(0.1, 0.2), self.dynamic_car_state._rear_axle_acceleration_2d)
        self.assertEqual(2, self.dynamic_car_state._angular_velocity)
        self.assertEqual(2.5, self.dynamic_car_state._angular_acceleration)
        self.assertEqual(0.5, self.dynamic_car_state._tire_steering_rate)

    def test_properties(self) -> None:
        """Checks that the properties return the expected variables."""
        self.assertTrue(self.dynamic_car_state.rear_axle_velocity_2d is self.dynamic_car_state._rear_axle_velocity_2d)
        self.assertTrue(
            self.dynamic_car_state.rear_axle_acceleration_2d is self.dynamic_car_state._rear_axle_acceleration_2d
        )
        self.assertTrue(self.dynamic_car_state.tire_steering_rate is self.dynamic_car_state._tire_steering_rate)
        self.assertTrue(self.dynamic_car_state.tire_steering_rate is self.dynamic_car_state._tire_steering_rate)

        self.assertAlmostEqual(4.104875150354758, self.dynamic_car_state.speed)
        self.assertEqual(0.22360679774997896, self.dynamic_car_state.acceleration)

    @patch('nuplan.common.actor_state.dynamic_car_state.StateVector2D', Mock())
    @patch("nuplan.common.actor_state.dynamic_car_state.DynamicCarState", autospec=DynamicCarState)
    def test_build_from_rear_axle(self, mock_dynamic_car_state: Mock) -> None:
        """Tests that constructor from rear axle behaves as intended."""
        mock_velocity = Mock()
        mock_acceleration = Mock()
        self.dynamic_car_state.build_from_rear_axle(1, mock_velocity, mock_acceleration, 4, 5, 6)
        mock_dynamic_car_state.assert_called_with(
            rear_axle_to_center_dist=1,
            rear_axle_velocity_2d=mock_velocity,
            rear_axle_acceleration_2d=mock_acceleration,
            angular_velocity=4,
            angular_acceleration=5,
            tire_steering_rate=6,
        )

    @patch('nuplan.common.actor_state.dynamic_car_state.StateVector2D')
    @patch('nuplan.common.actor_state.dynamic_car_state.math', Mock())
    @patch('nuplan.common.actor_state.dynamic_car_state._angular_velocity_from_cog')
    @patch('nuplan.common.actor_state.dynamic_car_state._projected_velocities_from_cog')
    @patch('nuplan.common.actor_state.dynamic_car_state._project_accelerations_from_cog')
    @patch('nuplan.common.actor_state.dynamic_car_state._get_beta')
    @patch("nuplan.common.actor_state.dynamic_car_state.DynamicCarState", autospec=DynamicCarState)
    def test_build_from_cog(
        self,
        mock_dynamic_car_state: Mock,
        mock_beta: Mock,
        mock_accelerations: Mock,
        mock_velocities: Mock,
        mock_angular_velocity: Mock,
        mock_vector: Mock,
    ) -> None:
        """Checks that constructor from COG computes the correct projections."""
        wheel_base = MagicMock(return_value="wheel_base")
        rear_axle_to_center = MagicMock(return_value="rear_axle_to_center")
        cog_speed = MagicMock(return_value="cog_speed")
        cog_acceleration = MagicMock(return_value="cog_acceleration")
        steering_angle = MagicMock(return_value="steering_angle")
        angular_accel = MagicMock(return_value="angular_accel")
        tire_steering_rate = MagicMock(return_value="tire_steering_rate")

        mock_velocities.return_value = ("x_vel", "y_vel")
        mock_accelerations.return_value = ("x_acc", "y_acc")

        self.dynamic_car_state.build_from_cog(
            wheel_base,
            rear_axle_to_center,
            cog_speed,
            cog_acceleration,
            steering_angle,
            angular_accel,
            tire_steering_rate,
        )

        mock_beta.assert_called_once_with(steering_angle, wheel_base)

        mock_velocities.assert_called_once_with(mock_beta.return_value, cog_speed)
        mock_angular_velocity.assert_called_once_with(cog_speed, wheel_base, mock_beta.return_value, steering_angle)
        mock_accelerations.assert_called_once_with(
            "x_vel", mock_angular_velocity.return_value, cog_acceleration, mock_beta.return_value
        )

        mock_dynamic_car_state.assert_called_with(
            rear_axle_to_center_dist=rear_axle_to_center,
            rear_axle_velocity_2d=mock_vector(mock_velocities.return_value),
            rear_axle_acceleration_2d=mock_vector(mock_accelerations.return_value),
            angular_velocity=mock_angular_velocity.return_value,
            angular_acceleration=angular_accel,
            tire_steering_rate=tire_steering_rate,
        )


if __name__ == '__main__':
    unittest.main()
