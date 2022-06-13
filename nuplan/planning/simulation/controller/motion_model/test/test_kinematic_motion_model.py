import unittest

import numpy as np

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.test.test_utils import get_sample_ego_state
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
    KinematicBicycleModel,
    forward_integrate,
)


class TestKinematicMotionModel(unittest.TestCase):
    """
    Run tests for Kinematic Bicycle Model.
    """

    def setUp(self) -> None:
        """Inherited, see superclass."""
        self.vehicle = get_pacifica_parameters()
        self.ego_state = get_sample_ego_state()
        self.sampling_time = 1.0
        self.motion_model = KinematicBicycleModel(self.vehicle)

        wheel_base = self.vehicle.wheel_base

        self.longitudinal_speed = self.ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        self.x_dot = self.longitudinal_speed * np.cos(self.ego_state.rear_axle.heading)
        self.y_dot = self.longitudinal_speed * np.sin(self.ego_state.rear_axle.heading)
        self.yaw_dot = self.longitudinal_speed * np.tan(self.ego_state.tire_steering_angle) / wheel_base

    def test_get_state_dot(self) -> None:
        """
        Test get_state_dot for expected results
        """
        state_dot = self.motion_model.get_state_dot(self.ego_state)
        self.assertEqual(state_dot.rear_axle, StateSE2(self.x_dot, self.y_dot, self.yaw_dot))
        self.assertEqual(
            state_dot.dynamic_car_state.rear_axle_velocity_2d,
            self.ego_state.dynamic_car_state.rear_axle_acceleration_2d,
        )
        self.assertEqual(state_dot.dynamic_car_state.rear_axle_acceleration_2d, StateVector2D(0, 0))
        self.assertEqual(state_dot.tire_steering_angle, self.ego_state.dynamic_car_state.tire_steering_rate)

    def test_propagate_state(self) -> None:
        """
        Test propagate_state
        """
        state = self.motion_model.propagate_state(self.ego_state, self.sampling_time)
        self.assertEqual(
            state.rear_axle,
            StateSE2(
                forward_integrate(self.ego_state.rear_axle.x, self.x_dot, self.sampling_time),
                forward_integrate(self.ego_state.rear_axle.y, self.y_dot, self.sampling_time),
                forward_integrate(self.ego_state.rear_axle.heading, self.yaw_dot, self.sampling_time),
            ),
        )
        self.assertEqual(
            state.dynamic_car_state.rear_axle_velocity_2d,
            StateVector2D(
                forward_integrate(
                    self.ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                    self.ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
                    self.sampling_time,
                ),
                0.0,
            ),
        )
        self.assertEqual(state.dynamic_car_state.rear_axle_acceleration_2d, StateVector2D(0.0, 0.0))
        self.assertEqual(
            state.tire_steering_angle,
            forward_integrate(
                self.ego_state.tire_steering_angle,
                self.ego_state.dynamic_car_state.tire_steering_rate,
                self.sampling_time,
            ),
        )
        self.assertEqual(
            state.dynamic_car_state.angular_velocity,
            state.dynamic_car_state.rear_axle_velocity_2d.x
            * np.tan(state.tire_steering_angle)
            / self.vehicle.wheel_base,
        )

    def test_limit_steering_angle(self) -> None:
        """
        Test whether the KinematicBicycleModel correct enforces steering angle
        limits.
        """
        dynamic_car_state = DynamicCarState.build_from_rear_axle(
            self.vehicle.rear_axle_to_center,
            rear_axle_velocity_2d=StateVector2D(0.0, 0.0),
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
            tire_steering_rate=10.0,
        )
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=StateSE2(x=0.0, y=0.0, heading=0.0), vehicle_parameters=self.vehicle
        )
        ego_state = EgoState(
            car_footprint,
            dynamic_car_state,
            tire_steering_angle=self.motion_model.max_steering_angle - 1e-4,
            is_in_auto_mode=True,
            time_point=TimePoint(0.0),
        )

        propagated_state = self.motion_model.propagate_state(ego_state, self.sampling_time)
        self.assertEqual(propagated_state.tire_steering_angle, self.motion_model.max_steering_angle)


if __name__ == '__main__':
    unittest.main()
