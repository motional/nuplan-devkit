import unittest

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState, get_acceleration_shifted, get_velocity_shifted
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.common.actor_state.test.test_utils import get_sample_dynamic_car_state, get_sample_ego_state


class TestEgoState(unittest.TestCase):
    def setUp(self) -> None:
        self.ego_state = get_sample_ego_state()
        self.dynamic_car_state = get_sample_dynamic_car_state()

    def test_ego_state_extended_construction(self) -> None:
        """ Tests that the ego state extended can be constructed from a pre-existing ego state. """
        ego_state_ext = EgoState.from_raw_params(self.ego_state.rear_axle,
                                                 self.dynamic_car_state.rear_axle_velocity_2d,
                                                 self.dynamic_car_state.rear_axle_acceleration_2d,
                                                 self.ego_state.tire_steering_angle,
                                                 self.ego_state.time_point,
                                                 self.dynamic_car_state.angular_velocity,
                                                 self.dynamic_car_state.angular_acceleration)

        self.assertTrue(ego_state_ext.dynamic_car_state == self.dynamic_car_state)
        self.assertTrue(ego_state_ext.center == self.ego_state.center)

    def test_cast_to_agent(self) -> None:
        """ Tests that the ego state extended can be cast to an Agent object. """
        ego_state_ext = EgoState.from_raw_params(self.ego_state.rear_axle,
                                                 self.dynamic_car_state.rear_axle_velocity_2d,
                                                 self.dynamic_car_state.rear_axle_acceleration_2d,
                                                 self.ego_state.tire_steering_angle,
                                                 self.ego_state.time_point,
                                                 self.dynamic_car_state.angular_velocity,
                                                 self.dynamic_car_state.angular_acceleration)
        ego_agent = ego_state_ext.agent
        self.assertEqual("ego", ego_agent.token)
        self.assertTrue(ego_state_ext.car_footprint.oriented_box is ego_agent.box)
        self.assertTrue(ego_state_ext.dynamic_car_state.center_velocity_2d is ego_agent.velocity)


class TestKinematicTransferFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.displacement = StateVector2D(2.0, 2.0)
        self.reference_vector = StateVector2D(2.3, 3.4)  # Can be used for both velocity and acceleration
        self.angular_velocity = 0.2

    def test_velocity_transfer(self) -> None:
        """ Tests behavior of velocity transfer formula for planar rigid bodies. """
        # Nominal case
        actual_velocity = get_velocity_shifted(self.displacement, self.reference_vector, self.angular_velocity)
        expected_velocity_p2 = StateVector2D(1.9, 3.8)
        np.testing.assert_array_almost_equal(expected_velocity_p2.array, actual_velocity.array, 6)  # type: ignore

        # No displacement
        actual_velocity = get_velocity_shifted(StateVector2D(0.0, 0.0), self.reference_vector, self.angular_velocity)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_velocity.array, 6)  # type: ignore

        # No rotation
        actual_velocity = get_velocity_shifted(self.displacement, self.reference_vector, 0)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_velocity.array, 6)  # type: ignore

    def test_acceleration_transfer(self) -> None:
        """ Tests behavior of acceleration transfer formula for planar rigid bodies. """
        # Nominal case
        angular_acceleration = 0.234
        actual_acceleration = get_acceleration_shifted(self.displacement, self.reference_vector, self.angular_velocity,
                                                       angular_acceleration)
        np.testing.assert_array_almost_equal(StateVector2D(2.848, 3.948).array,  # type: ignore
                                             actual_acceleration.array, 6)

        # No displacement
        actual_acceleration = get_acceleration_shifted(StateVector2D(0.0, 0.0), self.reference_vector,
                                                       self.angular_velocity, angular_acceleration)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_acceleration.array, 6)  # type: ignore

        # No rotation and acceleration
        actual_acceleration = get_acceleration_shifted(self.displacement, self.reference_vector, 0, 0)
        np.testing.assert_array_almost_equal(self.reference_vector.array, actual_acceleration.array, 6)  # type: ignore


if __name__ == '__main__':
    unittest.main()
