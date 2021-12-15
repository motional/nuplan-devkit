from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
from nuplan.common.actor_state.agent import Agent, AgentType
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.utils import lazy_property


def get_velocity_shifted(displacement: StateVector2D, ref_velocity: StateVector2D,
                         ref_angular_vel: float) -> StateVector2D:
    """
    Computes the velocity at a query point on the same planar rigid body as a reference point.
    :param displacement: The displacement vector from the reference to the query point
    :param ref_velocity: The velocity vector at the reference point
    :param ref_angular_vel: The angular velocity of the body around the vertical axis
    :return: The velocity vector at the given displacement.
    """
    # From cross product of velocity transfer formula in 2D
    velocity_shift_term = np.array([-displacement.y * ref_angular_vel, displacement.x * ref_angular_vel])
    return StateVector2D(*(ref_velocity.array + velocity_shift_term))


def get_acceleration_shifted(displacement: StateVector2D, ref_accel: StateVector2D, ref_angular_vel: float,
                             ref_angular_accel: float) -> StateVector2D:
    """
    Computes the acceleration at a query point on the same planar rigid body as a reference point.
    :param displacement: The displacement vector from the reference to the query point
    :param ref_accel: The acceleration vector at the reference point
    :param ref_angular_vel: The angular velocity of the body around the vertical axis
    :param ref_angular_accel: The angular acceleration of the body around the vertical axis
    :return: The acceleration vector at the given displacement.
    """
    centripetal_acceleration_term = displacement.array * ref_angular_vel ** 2
    angular_acceleration_term = displacement.array * ref_angular_accel

    return StateVector2D(*(ref_accel.array + centripetal_acceleration_term + angular_acceleration_term))


@dataclass
class DynamicCarState:
    """ Contains the various dynamic attributes of ego. """

    def __init__(self,
                 rear_axle_to_center_dist: float,
                 rear_axle_velocity_2d: Optional[StateVector2D] = None,
                 rear_axle_acceleration_2d: Optional[StateVector2D] = None,
                 center_velocity_2d: Optional[StateVector2D] = None,
                 center_acceleration_2d: Optional[StateVector2D] = None,
                 angular_velocity: float = 0.0,
                 angular_acceleration: float = 0.0):
        """
        :param rear_axle_to_center_dist: Distance (positive) from rear axle to the geometrical center of ego
        :param rear_axle_velocity_2d: Velocity vector at the rear axle
        :param rear_axle_acceleration_2d: Acceleration vector at the rear axle
        :param angular_velocity: Angular velocity of ego
        :param angular_acceleration: Angular acceleration of ego
        """

        self._angular_velocity = angular_velocity
        self._angular_acceleration = angular_acceleration
        displacement = StateVector2D(rear_axle_to_center_dist, 0.0)

        if rear_axle_velocity_2d and rear_axle_acceleration_2d:
            self._rear_axle_velocity_2d = rear_axle_velocity_2d
            self._rear_axle_acceleration_2d = rear_axle_acceleration_2d
            self._center_velocity_2d = get_velocity_shifted(displacement, self._rear_axle_velocity_2d,
                                                            self._angular_velocity)
            self._center_acceleration_2d = get_acceleration_shifted(displacement, self._rear_axle_acceleration_2d,
                                                                    self._angular_velocity, self._angular_acceleration)
        elif center_velocity_2d and center_acceleration_2d:
            self._center_velocity_2d = center_velocity_2d
            self._center_acceleration_2d = center_acceleration_2d
            self._rear_axle_velocity_2d = get_velocity_shifted(displacement, self._center_velocity_2d,
                                                               self._angular_velocity)
            self._rear_axle_acceleration_2d = get_acceleration_shifted(displacement, self._center_acceleration_2d,
                                                                       self._angular_velocity,
                                                                       self._angular_acceleration)
        else:
            raise RuntimeError("Velocity and acceleration at the rear axle or at the COG required!")

        self._speed = np.hypot(self._center_velocity_2d.x, self._center_velocity_2d.y)
        self._acceleration = np.hypot(self._center_acceleration_2d.x, self._center_acceleration_2d.y)

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

    @property
    def center_velocity_2d(self) -> StateVector2D:
        """
        Returns the vectorial velocity at the geometrical center of Ego.
        :return: StateVector2D Containing the velocity at the geometrical center of Ego
        """
        return self._center_velocity_2d

    @property
    def center_acceleration_2d(self) -> StateVector2D:
        """
        Returns the vectorial acceleration at the geometrical center of Ego.
        :return: StateVector2D Containing the acceleration at the geometrical center of Ego
        """
        return self._center_acceleration_2d

    @property
    def angular_velocity(self) -> float:
        """
        Getter for the angular velocity of ego.
        :return: Angular velocity
        """
        return self._angular_velocity

    @property
    def angular_acceleration(self) -> float:
        """
        Getter for the angular acceleration of ego.
        :return: Angular acceleration
        """
        return self._angular_acceleration

    @property
    def speed(self) -> float:
        """
        Magnitude of the speed of the center of ego.
        :return: [m/s] 1D speed
        """
        return float(self._speed)

    @property
    def acceleration(self) -> float:
        """
        Magnitude of the acceleration of the center of ego.
        :return: [m/s^2] 1D acceleration
        """
        return float(self._acceleration)


class EgoState:
    """ Represent the current state of ego, along with its dynamic attributes. """

    def __init__(self, car_footprint: CarFootprint, dynamic_car_state: DynamicCarState, tire_steering_angle: float,
                 time_point: TimePoint):
        """
        :param car_footprint: The CarFootprint of Ego
        :param dynamic_car_state: The current dynamical state of ego
        :param tire_steering_angle: The current steering angle of the tires
        :param time_point: Time stamp of the state
        """
        self._car_footprint = car_footprint
        self._tire_steering_angle = tire_steering_angle
        self._time_point = time_point
        self._dynamic_car_state = dynamic_car_state

    def __iter__(self) -> Iterable[Union[int, float]]:
        return iter(
            (self.time_us, self.rear_axle.x, self.rear_axle.y, self.rear_axle.heading,
             self.dynamic_car_state.rear_axle_velocity_2d.x, self.dynamic_car_state.rear_axle_velocity_2d.y,
             self.dynamic_car_state.rear_axle_acceleration_2d.x, self.dynamic_car_state.rear_axle_acceleration_2d.y,
             self.tire_steering_angle)
        )

    @property
    def car_footprint(self) -> CarFootprint:
        """
        Getter for Ego's Car footprint
        :return: Ego's car footprint
        """
        return self._car_footprint

    @property
    def tire_steering_angle(self) -> float:
        """
        Getter for Ego's tire steering angle
        :return: Ego's tire steering angle
        """
        return self._tire_steering_angle

    @property
    def center(self) -> StateSE2:
        """
        Getter for Ego's center pose (center of mass)
        :return: Ego's center pose
        """
        return self._car_footprint.oriented_box.center

    @property
    def rear_axle(self) -> StateSE2:
        """
        Getter for Ego's rear axle pose (middle of the rear axle)
        :return: Ego's rear axle pose
        """
        return self.car_footprint.rear_axle

    @property
    def time_point(self) -> TimePoint:
        """
        Time stamp of the EgoState
        :return: EgoState time stamp
        """
        return self._time_point

    @property
    def time_us(self) -> int:
        """
        Time in micro seconds
        :return: [us]
        """
        return int(self.time_point.time_us)

    @property
    def time_seconds(self) -> float:
        """
        Time in seconds
        :return: [s]
        """
        return float(self.time_us * 1e-6)

    @classmethod
    def from_raw_params(cls, pose: StateSE2, velocity_2d: StateVector2D, acceleration_2d: StateVector2D,
                        tire_steering_angle: float, time_point: TimePoint, angular_vel: float = 0.0,
                        angular_accel: float = 0.0) -> EgoState:
        """
        Initializer using raw parameters, assumes that the reference frame is CAR_POINT.REAR_AXLE
        :param pose: Pose of ego's rear axle
        :param velocity_2d: Vectorial velocity of Ego's rear axle
        :param acceleration_2d: Vectorial acceleration of Ego's rear axle
        :param angular_vel: Angular velocity of Ego
        :param angular_accel: Angular acceleration of Ego,
        :param tire_steering_angle: Angle of the tires
        :param time_point: Timestamp of the ego state
        :return: The initialized EgoState
        """
        car_footprint = CarFootprint(pose)
        dynamic_ego_state = DynamicCarState(car_footprint.rear_axle_to_center_dist,
                                            rear_axle_velocity_2d=velocity_2d,
                                            rear_axle_acceleration_2d=acceleration_2d,
                                            angular_velocity=angular_vel,
                                            angular_acceleration=angular_accel)
        return cls(car_footprint, dynamic_ego_state, tire_steering_angle, time_point)

    @property
    def dynamic_car_state(self) -> DynamicCarState:
        """
        Getter for the dynamic car state of Ego.
        :return: The dynamic car state
        """
        return self._dynamic_car_state

    def _to_agent(self) -> Agent:
        """
        Casts EgoState to an agent object.
        :return: An Agent object with the parameters of EgoState"""
        return Agent(token="ego", agent_type=AgentType.EGO, oriented_box=self.car_footprint.oriented_box,
                     velocity=self.dynamic_car_state.center_velocity_2d)

    @lazy_property
    def agent(self) -> Agent:
        """
        Casts the EgoState to an Agent object.
        :return: An Agent
        """
        return self._to_agent()

    @staticmethod
    def deserialize(vector: List[Union[int, float]]) -> EgoState:
        """ Deserialize object, ordering kept for backward compatibility"""
        if len(vector) != 9:
            raise RuntimeError(f'Expected a vector of size 9, got {len(vector)}')
        return EgoState.from_raw_params(StateSE2(vector[1], vector[2], vector[3]),
                                        velocity_2d=StateVector2D(vector[4], vector[5]),
                                        acceleration_2d=StateVector2D(vector[6], vector[7]),
                                        tire_steering_angle=vector[8],
                                        time_point=TimePoint(int(vector[0]))
                                        )
