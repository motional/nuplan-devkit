import unittest

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.test.test_utils import get_sample_dynamic_car_state, get_sample_ego_state
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.utils.split_state import SplitState


class TestEgoState(unittest.TestCase):
    """Tests EgoState class"""

    def setUp(self) -> None:
        """Creates sample parameters for testing"""
        self.ego_state = get_sample_ego_state()
        self.vehicle = get_pacifica_parameters()
        self.dynamic_car_state = get_sample_dynamic_car_state(self.vehicle.rear_axle_to_center)

    def test_ego_state_extended_construction(self) -> None:
        """Tests that the ego state extended can be constructed from a pre-existing ego state."""
        ego_state_ext = EgoState.build_from_rear_axle(
            rear_axle_pose=self.ego_state.rear_axle,
            rear_axle_velocity_2d=self.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=self.dynamic_car_state.rear_axle_acceleration_2d,
            tire_steering_angle=self.ego_state.tire_steering_angle,
            time_point=self.ego_state.time_point,
            angular_vel=self.dynamic_car_state.angular_velocity,
            angular_accel=self.dynamic_car_state.angular_acceleration,
            is_in_auto_mode=True,
            vehicle_parameters=self.vehicle,
        )

        self.assertTrue(ego_state_ext.dynamic_car_state == self.dynamic_car_state)
        self.assertTrue(ego_state_ext.center == self.ego_state.center)

        # Test Waypoint
        wp = ego_state_ext.waypoint
        self.assertEqual(wp.time_point, ego_state_ext.time_point)
        self.assertEqual(wp.oriented_box, ego_state_ext.car_footprint)
        self.assertEqual(wp.velocity, ego_state_ext.dynamic_car_state.rear_axle_velocity_2d)

    def test_to_split_state(self) -> None:
        """Tests that the state gets split as expected"""
        split_state = self.ego_state.to_split_state()
        self.assertEqual(len(split_state.linear_states), 8)
        self.assertEqual(split_state.fixed_states, [self.ego_state.car_footprint.vehicle_parameters])
        self.assertEqual(split_state.angular_states, [self.ego_state.rear_axle.heading])

    def test_from_split_state(self) -> None:
        """Tests that the object gets created as expected from the split state"""
        split_state = SplitState([0, 1, 2, 3, 4, 5, 6, 7], [8], [self.ego_state.car_footprint.vehicle_parameters])

        ego_from_split = EgoState.from_split_state(split_state)

        self.assertEqual(
            self.ego_state.car_footprint.vehicle_parameters, ego_from_split.car_footprint.vehicle_parameters
        )
        self.assertAlmostEqual(ego_from_split.time_us, 0)
        self.assertAlmostEqual(ego_from_split.rear_axle.x, 1)
        self.assertAlmostEqual(ego_from_split.rear_axle.y, 2)
        self.assertAlmostEqual(ego_from_split.rear_axle.heading, 8)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_velocity_2d.x, 3)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_velocity_2d.y, 4)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_acceleration_2d.x, 5)
        self.assertAlmostEqual(ego_from_split.dynamic_car_state.rear_axle_acceleration_2d.y, 6)
        self.assertAlmostEqual(ego_from_split.tire_steering_angle, 7)


if __name__ == '__main__':
    unittest.main()
