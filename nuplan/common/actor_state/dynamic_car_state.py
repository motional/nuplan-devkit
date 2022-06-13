from __future__ import annotations

import math
from functools import cached_property
from typing import Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateVector2D


def get_velocity_shifted(
    displacement: StateVector2D, ref_velocity: StateVector2D, ref_angular_vel: float
) -> StateVector2D:
    """
    Computes the velocity at a query point on the same planar rigid body as a reference point.
    :param displacement: [m] The displacement vector from the reference to the query point
    :param ref_velocity: [m/s] The velocity vector at the reference point
    :param ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis
    :return: [m/s] The velocity vector at the given displacement.
    """
    # From cross product of velocity transfer formula in 2D
    velocity_shift_term: npt.NDArray[np.float64] = np.array(
        [-displacement.y * ref_angular_vel, displacement.x * ref_angular_vel]
    )
    return StateVector2D(*(ref_velocity.array + velocity_shift_term))


def get_acceleration_shifted(
    displacement: StateVector2D, ref_accel: StateVector2D, ref_angular_vel: float, ref_angular_accel: float
) -> StateVector2D:
    """
    Computes the acceleration at a query point on the same planar rigid body as a reference point.
    :param displacement: [m] The displacement vector from the reference to the query point
    :param ref_accel: [m/s^2] The acceleration vector at the reference point
    :param ref_angular_vel: [rad/s] The angular velocity of the body around the vertical axis
    :param ref_angular_accel: [rad/s^2] The angular acceleration of the body around the vertical axis
    :return: [m/s^2] The acceleration vector at the given displacement.
    """
    centripetal_acceleration_term = displacement.array * ref_angular_vel**2
    angular_acceleration_term = displacement.array * ref_angular_accel

    return StateVector2D(*(ref_accel.array + centripetal_acceleration_term + angular_acceleration_term))


def _get_beta(steering_angle: float, wheel_base: float) -> float:
    """
    Computes beta, the angle from rear axle to COG at instantaneous center of rotation
    :param [rad] steering_angle: steering angle of the car
    :param [m] wheel_base: distance between the axles
    :return: [rad] Value of beta
    """
    beta = math.atan2(math.tan(steering_angle), wheel_base)
    return beta


def _projected_velocities_from_cog(beta: float, cog_speed: float) -> Tuple[float, float]:
    """
    Computes the projected velocities at the rear axle using the Bicycle kinematic model using COG data
    :param beta: [rad] the angle from rear axle to COG at instantaneous center of rotation
    :param cog_speed: [m/s] Magnitude of velocity vector at COG
    :return: Tuple with longitudinal and lateral velocities [m/s] at the rear axle
    """
    # This gives COG longitudinal, which is the same as rear axle
    rear_axle_forward_velocity = math.cos(beta) * cog_speed  # [m/s]
    # Lateral velocity is zero, by model assumption
    rear_axle_lateral_velocity = 0

    return rear_axle_forward_velocity, rear_axle_lateral_velocity


def _angular_velocity_from_cog(
    cog_speed: float, length_rear_axle_to_cog: float, beta: float, steering_angle: float
) -> float:
    """
    Computes the angular velocity using the Bicycle kinematic model using COG data.
    :param cog_speed: [m/s] Magnitude of velocity vector at COG
    :param length_rear_axle_to_cog: [m] Distance from rear axle to COG
    :param beta: [rad] angle from rear axle to COG at instantaneous center of rotation
    :param steering_angle: [rad] of the car
    """
    return (cog_speed / length_rear_axle_to_cog) * math.cos(beta) * math.tan(steering_angle)


def _project_accelerations_from_cog(
    rear_axle_longitudinal_velocity: float, angular_velocity: float, cog_acceleration: float, beta: float
) -> Tuple[float, float]:
    """
    Computes the projected accelerations at the rear axle using the Bicycle kinematic model using COG data
    :param rear_axle_longitudinal_velocity: [m/s] Longitudinal component of velocity vector at COG
    :param angular_velocity: [rad/s] Angular velocity at COG
    :param cog_acceleration: [m/s^2] Magnitude of acceleration vector at COG
    :param beta: [rad] ]the angle from rear axle to COG at instantaneous center of rotation
    :return: Tuple with longitudinal and lateral velocities [m/s] at the rear axle
    """
    # Rigid body assumption, can project from COG
    rear_axle_longitudinal_acceleration = math.cos(beta) * cog_acceleration  # [m/s^2]

    # Centripetal accel is a=v^2 / R and angular_velocity = v / R
    rear_axle_lateral_acceleration = rear_axle_longitudinal_velocity * angular_velocity  # [m/s^2]

    return rear_axle_longitudinal_acceleration, rear_axle_lateral_acceleration


class DynamicCarState:
    """Contains the various dynamic attributes of ego."""

    def __init__(
        self,
        rear_axle_to_center_dist: float,
        rear_axle_velocity_2d: StateVector2D,
        rear_axle_acceleration_2d: StateVector2D,
        angular_velocity: float = 0.0,
        angular_acceleration: float = 0.0,
        tire_steering_rate: float = 0.0,
    ):
        """
        :param rear_axle_to_center_dist:[m]  Distance (positive) from rear axle to the geometrical center of ego
        :param rear_axle_velocity_2d: [m/s]Velocity vector at the rear axle
        :param rear_axle_acceleration_2d: [m/s^2] Acceleration vector at the rear axle
        :param angular_velocity: [rad/s] Angular velocity of ego
        :param angular_acceleration: [rad/s^2] Angular acceleration of ego
        :param tire_steering_rate: [rad/s] Tire steering rate of ego
        """
        self._rear_axle_to_center_dist = rear_axle_to_center_dist
        self._angular_velocity = angular_velocity
        self._angular_acceleration = angular_acceleration
        self._rear_axle_velocity_2d = rear_axle_velocity_2d
        self._rear_axle_acceleration_2d = rear_axle_acceleration_2d
        self._tire_steering_rate = tire_steering_rate

    @property
    def rear_axle_velocity_2d(self) -> StateVector2D:
        """
        Returns the vectorial velocity at the middle of the rear axle.
        :return: StateVector2D Containing the velocity at the rear axle
        """
        return self._rear_axle_velocity_2d

    @property
    def rear_axle_acceleration_2d(self) -> StateVector2D:
        """
        Returns the vectorial acceleration at the middle of the rear axle.
        :return: StateVector2D Containing the acceleration at the rear axle
        """
        return self._rear_axle_acceleration_2d

    @cached_property
    def center_velocity_2d(self) -> StateVector2D:
        """
        Returns the vectorial velocity at the geometrical center of Ego.
        :return: StateVector2D Containing the velocity at the geometrical center of Ego
        """
        displacement = StateVector2D(self._rear_axle_to_center_dist, 0.0)
        return get_velocity_shifted(displacement, self.rear_axle_velocity_2d, self.angular_velocity)

    @cached_property
    def center_acceleration_2d(self) -> StateVector2D:
        """
        Returns the vectorial acceleration at the geometrical center of Ego.
        :return: StateVector2D Containing the acceleration at the geometrical center of Ego
        """
        displacement = StateVector2D(self._rear_axle_to_center_dist, 0.0)
        return get_acceleration_shifted(
            displacement, self.rear_axle_acceleration_2d, self.angular_velocity, self.angular_acceleration
        )

    @property
    def angular_velocity(self) -> float:
        """
        Getter for the angular velocity of ego.
        :return: [rad/s] Angular velocity
        """
        return self._angular_velocity

    @property
    def angular_acceleration(self) -> float:
        """
        Getter for the angular acceleration of ego.
        :return: [rad/s^2] Angular acceleration
        """
        return self._angular_acceleration

    @property
    def tire_steering_rate(self) -> float:
        """
        Getter for the tire steering rate of ego.
        :return: [rad/s] Tire steering rate
        """
        return self._tire_steering_rate

    @cached_property
    def speed(self) -> float:
        """
        Magnitude of the speed of the center of ego.
        :return: [m/s] 1D speed
        """
        return float(self._rear_axle_velocity_2d.magnitude())

    @cached_property
    def acceleration(self) -> float:
        """
        Magnitude of the acceleration of the center of ego.
        :return: [m/s^2] 1D acceleration
        """
        return float(self._rear_axle_acceleration_2d.magnitude())

    def __eq__(self, other: object) -> bool:
        """
        Compare two instances whether they are numerically close
        :param other: object
        :return: true if the classes are almost equal
        """
        if not isinstance(other, DynamicCarState):
            # Return NotImplemented in case the classes do not match
            return NotImplemented

        return (
            self.rear_axle_velocity_2d == other.rear_axle_velocity_2d
            and self.rear_axle_acceleration_2d == other.rear_axle_acceleration_2d
            and math.isclose(self._angular_acceleration, other._angular_acceleration)
            and math.isclose(self._angular_velocity, other._angular_velocity)
            and math.isclose(self._rear_axle_to_center_dist, other._rear_axle_to_center_dist)
            and math.isclose(self._tire_steering_rate, other._tire_steering_rate)
        )

    def __repr__(self) -> str:
        """Repr magic method"""
        return (
            f"Rear Axle| velocity: {self.rear_axle_velocity_2d}, acceleration: {self.rear_axle_acceleration_2d}\n"
            f"Center   | velocity: {self.center_velocity_2d}, acceleration: {self.center_acceleration_2d}\n"
            f"angular velocity: {self.angular_velocity}, angular acceleration: {self._angular_acceleration}\n"
            f"rear_axle_to_center_dist: {self._rear_axle_to_center_dist} \n"
            f"_tire_steering_rate: {self._tire_steering_rate} \n"
        )

    @staticmethod
    def build_from_rear_axle(
        rear_axle_to_center_dist: float,
        rear_axle_velocity_2d: StateVector2D,
        rear_axle_acceleration_2d: StateVector2D,
        angular_velocity: float = 0.0,
        angular_acceleration: float = 0.0,
        tire_steering_rate: float = 0.0,
    ) -> DynamicCarState:
        """
        Construct ego state from rear axle parameters
        :param rear_axle_to_center_dist: [m] distance between center and rear axle
        :param rear_axle_velocity_2d: [m/s] velocity at rear axle
        :param rear_axle_acceleration_2d: [m/s^2] acceleration at rear axle
        :param angular_velocity: [rad/s] angular velocity
        :param angular_acceleration: [rad/s^2] angular acceleration
        :param tire_steering_rate: [rad/s] tire steering_rate
        :return: constructed DynamicCarState of ego.
        """
        return DynamicCarState(
            rear_axle_to_center_dist=rear_axle_to_center_dist,
            rear_axle_velocity_2d=rear_axle_velocity_2d,
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            tire_steering_rate=tire_steering_rate,
        )

    @staticmethod
    def build_from_cog(
        wheel_base: float,
        rear_axle_to_center_dist: float,
        cog_speed: float,
        cog_acceleration: float,
        steering_angle: float,
        angular_acceleration: float = 0.0,
        tire_steering_rate: float = 0.0,
    ) -> DynamicCarState:
        """
        Construct ego state from rear axle parameters
        :param wheel_base: distance between axles [m]
        :param rear_axle_to_center_dist: distance between center and rear axle [m]
        :param cog_speed: magnitude of speed COG [m/s]
        :param cog_acceleration: magnitude of acceleration at COG [m/s^s]
        :param steering_angle: steering angle at tire [rad]
        :param angular_acceleration: angular acceleration
        :param tire_steering_rate: tire steering rate
        :return: constructed DynamicCarState of ego.
        """
        # under kinematic state assumption: compute additionally needed states
        beta = _get_beta(steering_angle, wheel_base)

        rear_axle_longitudinal_velocity, rear_axle_lateral_velocity = _projected_velocities_from_cog(beta, cog_speed)

        angular_velocity = _angular_velocity_from_cog(cog_speed, wheel_base, beta, steering_angle)

        # compute acceleration at rear axle given the kinematic assumptions
        longitudinal_acceleration, lateral_acceleration = _project_accelerations_from_cog(
            rear_axle_longitudinal_velocity, angular_velocity, cog_acceleration, beta
        )

        return DynamicCarState(
            rear_axle_to_center_dist=rear_axle_to_center_dist,
            rear_axle_velocity_2d=StateVector2D(rear_axle_longitudinal_velocity, rear_axle_lateral_velocity),
            rear_axle_acceleration_2d=StateVector2D(longitudinal_acceleration, lateral_acceleration),
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
            tire_steering_rate=tire_steering_rate,
        )
