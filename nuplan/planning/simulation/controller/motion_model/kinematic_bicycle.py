import numpy as np

from nuplan.common.actor_state.ego_state import EgoState, EgoStateDot
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.compute import principal_value
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel
from nuplan.planning.simulation.controller.utils import forward_integrate


class KinematicBicycleModel(AbstractMotionModel):
    """
    A class describing the kinematic motion model where the rear axle is the point of reference.
    """

    def __init__(self, vehicle: VehicleParameters, max_steering_angle: float = np.pi / 3):
        """
        Construct KinematicBicycleModel.

        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        """
        self.vehicle = vehicle
        self.max_steering_angle = max_steering_angle

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """Inherited, see super class."""
        longitudinal_speed = state.dynamic_car_state.rear_axle_velocity_2d.x
        x_dot = longitudinal_speed * np.cos(state.rear_axle.heading)
        y_dot = longitudinal_speed * np.sin(state.rear_axle.heading)
        yaw_dot = longitudinal_speed * np.tan(state.tire_steering_angle) / self.vehicle.wheel_base

        return EgoStateDot.build_from_rear_axle(
            rear_axle_pose=StateSE2(x=x_dot, y=y_dot, heading=yaw_dot),
            rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_acceleration_2d,
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
            tire_steering_angle=state.dynamic_car_state.tire_steering_rate,
            time_point=state.time_point,
            is_in_auto_mode=True,
            vehicle_parameters=self.vehicle,
        )

    def propagate_state(self, state: EgoState, sampling_time: float) -> EgoState:
        """Inherited, see super class."""
        # Compute state derivatives
        state_dot = self.get_state_dot(state)

        # Integrate position and heading
        next_x = forward_integrate(state.rear_axle.x, state_dot.rear_axle.x, sampling_time)
        next_y = forward_integrate(state.rear_axle.y, state_dot.rear_axle.y, sampling_time)
        next_heading = forward_integrate(state.rear_axle.heading, state_dot.rear_axle.heading, sampling_time)
        # Wrap angle between [-pi, pi]
        next_heading = principal_value(next_heading)

        # Compute rear axle velocity in car frame
        next_point_velocity_x = forward_integrate(
            state.dynamic_car_state.rear_axle_velocity_2d.x,
            state_dot.dynamic_car_state.rear_axle_velocity_2d.x,
            sampling_time,
        )
        next_point_velocity_y = 0.0  # Lateral velocity is always zero in kinematic bicycle model

        # Integrate steering angle and clip to bounds
        next_point_tire_steering_angle = np.clip(
            forward_integrate(state.tire_steering_angle, state_dot.tire_steering_angle, sampling_time),
            -self.max_steering_angle,
            self.max_steering_angle,
        )

        # Compute angular velocity
        next_point_angular_velocity = (
            next_point_velocity_x * np.tan(next_point_tire_steering_angle) / self.vehicle.wheel_base
        )

        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(next_x, next_y, next_heading),
            rear_axle_velocity_2d=StateVector2D(next_point_velocity_x, next_point_velocity_y),
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
            angular_vel=next_point_angular_velocity,
            tire_steering_angle=float(next_point_tire_steering_angle),
            time_point=state.time_point + TimePoint(int(sampling_time * 1e6)),
            is_in_auto_mode=True,
            vehicle_parameters=self.vehicle,
        )
