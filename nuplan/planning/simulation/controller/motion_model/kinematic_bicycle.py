import numpy as np

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
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

    def __init__(
        self,
        vehicle: VehicleParameters,
        max_steering_angle: float = np.pi / 3,
        accel_time_constant: float = 0.2,
        steering_angle_time_constant: float = 0.05,
    ):
        """
        Construct KinematicBicycleModel.

        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        :param accel_time_constant: low pass filter time constant for acceleration in s
        :param steering_angle_time_constant: low pass filter time constant for steering angle in s
        """
        self._vehicle = vehicle
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """Inherited, see super class."""
        longitudinal_speed = state.dynamic_car_state.rear_axle_velocity_2d.x
        x_dot = longitudinal_speed * np.cos(state.rear_axle.heading)
        y_dot = longitudinal_speed * np.sin(state.rear_axle.heading)
        yaw_dot = longitudinal_speed * np.tan(state.tire_steering_angle) / self._vehicle.wheel_base

        return EgoStateDot.build_from_rear_axle(
            rear_axle_pose=StateSE2(x=x_dot, y=y_dot, heading=yaw_dot),
            rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_acceleration_2d,
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
            tire_steering_angle=state.dynamic_car_state.tire_steering_rate,
            time_point=state.time_point,
            is_in_auto_mode=True,
            vehicle_parameters=self._vehicle,
        )

    def _update_commands(
        self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint
    ) -> EgoState:
        """
        This function applies some first order control delay/a low pass filter to acceleration/steering.

        :param state: Ego state
        :param ideal_dynamic_state: The desired dynamic state for propagation
        :param sampling_time: The time duration to propagate for
        :return: propagating_state including updated dynamic_state
        """
        dt_control = sampling_time.time_s
        accel = state.dynamic_car_state.rear_axle_acceleration_2d.x
        steering_angle = state.tire_steering_angle

        ideal_accel_x = ideal_dynamic_state.rear_axle_acceleration_2d.x
        ideal_steering_angle = dt_control * ideal_dynamic_state.tire_steering_rate + steering_angle

        updated_accel_x = dt_control / (dt_control + self._accel_time_constant) * (ideal_accel_x - accel) + accel
        updated_steering_angle = (
            dt_control / (dt_control + self._steering_angle_time_constant) * (ideal_steering_angle - steering_angle)
            + steering_angle
        )
        updated_steering_rate = (updated_steering_angle - steering_angle) / dt_control

        dynamic_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(updated_accel_x, 0),
            tire_steering_rate=updated_steering_rate,
        )
        propagating_state = EgoState(
            car_footprint=state.car_footprint,
            dynamic_car_state=dynamic_state,
            tire_steering_angle=state.tire_steering_angle,
            is_in_auto_mode=True,
            time_point=state.time_point,
        )
        return propagating_state

    def propagate_state(
        self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint
    ) -> EgoState:
        """Inherited, see super class."""
        propagating_state = self._update_commands(state, ideal_dynamic_state, sampling_time)

        # Compute state derivatives
        state_dot = self.get_state_dot(propagating_state)

        # Integrate position and heading
        next_x = forward_integrate(propagating_state.rear_axle.x, state_dot.rear_axle.x, sampling_time)
        next_y = forward_integrate(propagating_state.rear_axle.y, state_dot.rear_axle.y, sampling_time)
        next_heading = forward_integrate(
            propagating_state.rear_axle.heading, state_dot.rear_axle.heading, sampling_time
        )
        # Wrap angle between [-pi, pi]
        next_heading = principal_value(next_heading)

        # Compute rear axle velocity in car frame
        next_point_velocity_x = forward_integrate(
            propagating_state.dynamic_car_state.rear_axle_velocity_2d.x,
            state_dot.dynamic_car_state.rear_axle_velocity_2d.x,
            sampling_time,
        )
        next_point_velocity_y = 0.0  # Lateral velocity is always zero in kinematic bicycle model

        # Integrate steering angle and clip to bounds
        next_point_tire_steering_angle = np.clip(
            forward_integrate(propagating_state.tire_steering_angle, state_dot.tire_steering_angle, sampling_time),
            -self._max_steering_angle,
            self._max_steering_angle,
        )

        # Compute angular velocity
        next_point_angular_velocity = (
            next_point_velocity_x * np.tan(next_point_tire_steering_angle) / self._vehicle.wheel_base
        )

        rear_axle_accel = [
            state_dot.dynamic_car_state.rear_axle_velocity_2d.x,
            state_dot.dynamic_car_state.rear_axle_velocity_2d.y,
        ]
        angular_accel = (next_point_angular_velocity - state.dynamic_car_state.angular_velocity) / sampling_time.time_s

        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(next_x, next_y, next_heading),
            rear_axle_velocity_2d=StateVector2D(next_point_velocity_x, next_point_velocity_y),
            rear_axle_acceleration_2d=StateVector2D(rear_axle_accel[0], rear_axle_accel[1]),
            tire_steering_angle=float(next_point_tire_steering_angle),
            time_point=propagating_state.time_point + sampling_time,
            vehicle_parameters=self._vehicle,
            is_in_auto_mode=True,
            angular_vel=next_point_angular_velocity,
            angular_accel=angular_accel,
            tire_steering_rate=state_dot.tire_steering_angle,
        )
