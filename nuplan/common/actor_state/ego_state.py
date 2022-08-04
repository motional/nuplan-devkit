from __future__ import annotations

from functools import cached_property
from typing import Iterable, List, Union

from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState, get_acceleration_shifted, get_velocity_shifted
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.utils.split_state import SplitState


class EgoState(InterpolatableState):
    """Represent the current state of ego, along with its dynamic attributes."""

    def __init__(
        self,
        car_footprint: CarFootprint,
        dynamic_car_state: DynamicCarState,
        tire_steering_angle: float,
        is_in_auto_mode: bool,
        time_point: TimePoint,
    ):
        """
        :param car_footprint: The CarFootprint of Ego
        :param dynamic_car_state: The current dynamical state of ego
        :param tire_steering_angle: The current steering angle of the tires
        :param is_in_auto_mode: If the state refers to car in autonomous mode
        :param time_point: Time stamp of the state
        """
        self._car_footprint = car_footprint
        self._tire_steering_angle = tire_steering_angle
        self._is_in_auto_mode = is_in_auto_mode
        self._time_point = time_point
        self._dynamic_car_state = dynamic_car_state

    @cached_property
    def waypoint(self) -> Waypoint:
        """
        :return: waypoint corresponding to this ego state
        """
        return Waypoint(
            time_point=self.time_point,
            oriented_box=self.car_footprint,
            velocity=self.dynamic_car_state.rear_axle_velocity_2d,
        )

    @staticmethod
    def deserialize(vector: List[Union[int, float]], vehicle: VehicleParameters) -> EgoState:
        """
        Deserialize object, ordering kept for backward compatibility
        :param vector: List of variables for deserialization
        :param vehicle: Vehicle parameters
        """
        if len(vector) != 9:
            raise RuntimeError(f'Expected a vector of size 9, got {len(vector)}')

        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(vector[1], vector[2], vector[3]),
            rear_axle_velocity_2d=StateVector2D(vector[4], vector[5]),
            rear_axle_acceleration_2d=StateVector2D(vector[6], vector[7]),
            tire_steering_angle=vector[8],
            time_point=TimePoint(int(vector[0])),
            vehicle_parameters=vehicle,
        )

    def __iter__(self) -> Iterable[Union[int, float]]:
        """Iterable over ego parameters"""
        return iter(
            (
                self.time_us,
                self.rear_axle.x,
                self.rear_axle.y,
                self.rear_axle.heading,
                self.dynamic_car_state.rear_axle_velocity_2d.x,
                self.dynamic_car_state.rear_axle_velocity_2d.y,
                self.dynamic_car_state.rear_axle_acceleration_2d.x,
                self.dynamic_car_state.rear_axle_acceleration_2d.y,
                self.tire_steering_angle,
            )
        )

    def to_split_state(self) -> SplitState:
        """Inherited, see superclass."""
        linear_states = [
            self.time_us,
            self.rear_axle.x,
            self.rear_axle.y,
            self.dynamic_car_state.rear_axle_velocity_2d.x,
            self.dynamic_car_state.rear_axle_velocity_2d.y,
            self.dynamic_car_state.rear_axle_acceleration_2d.x,
            self.dynamic_car_state.rear_axle_acceleration_2d.y,
            self.tire_steering_angle,
        ]
        angular_states = [self.rear_axle.heading]
        fixed_state = [self.car_footprint.vehicle_parameters]

        return SplitState(linear_states, angular_states, fixed_state)

    @staticmethod
    def from_split_state(split_state: SplitState) -> EgoState:
        """Inherited, see superclass."""
        if len(split_state) != 10:
            raise RuntimeError(f'Expected a variable state vector of size 10, got {len(split_state)}')

        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(
                split_state.linear_states[1], split_state.linear_states[2], split_state.angular_states[0]
            ),
            rear_axle_velocity_2d=StateVector2D(split_state.linear_states[3], split_state.linear_states[4]),
            rear_axle_acceleration_2d=StateVector2D(split_state.linear_states[5], split_state.linear_states[6]),
            tire_steering_angle=split_state.linear_states[7],
            time_point=TimePoint(int(split_state.linear_states[0])),
            vehicle_parameters=split_state.fixed_states[0],
        )

    @property
    def is_in_auto_mode(self) -> bool:
        """
        :return: True if ego is in auto mode, False otherwise.
        """
        return self._is_in_auto_mode

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
        :return: [us].
        """
        return int(self.time_point.time_us)

    @property
    def time_seconds(self) -> float:
        """
        Time in seconds
        :return: [s]
        """
        return float(self.time_us * 1e-6)

    @property
    def dynamic_car_state(self) -> DynamicCarState:
        """
        Getter for the dynamic car state of Ego.
        :return: The dynamic car state
        """
        return self._dynamic_car_state

    @property
    def scene_object_metadata(self) -> SceneObjectMetadata:
        """
        :return: create scene object metadata
        """
        return SceneObjectMetadata(token='ego', track_token="ego", track_id=-1, timestamp_us=self.time_us)

    @cached_property
    def agent(self) -> AgentState:
        """
        Casts the EgoState to an Agent object.
        :return: An Agent object with the parameters of EgoState
        """
        return AgentState(
            metadata=self.scene_object_metadata,
            tracked_object_type=TrackedObjectType.EGO,
            oriented_box=self.car_footprint.oriented_box,
            velocity=self.dynamic_car_state.center_velocity_2d,
        )

    @classmethod
    def build_from_rear_axle(
        cls,
        rear_axle_pose: StateSE2,
        rear_axle_velocity_2d: StateVector2D,
        rear_axle_acceleration_2d: StateVector2D,
        tire_steering_angle: float,
        time_point: TimePoint,
        vehicle_parameters: VehicleParameters,
        is_in_auto_mode: bool = True,
        angular_vel: float = 0.0,
        angular_accel: float = 0.0,
        tire_steering_rate: float = 0.0,
    ) -> EgoState:
        """
        Initializer using raw parameters, assumes that the reference frame is CAR_POINT.REAR_AXLE
        :param rear_axle_pose: Pose of ego's rear axle
        :param rear_axle_velocity_2d: Vectorial velocity of Ego's rear axle
        :param rear_axle_acceleration_2d: Vectorial acceleration of Ego's rear axle
        :param angular_vel: Angular velocity of Ego
        :param angular_accel: Angular acceleration of Ego,
        :param tire_steering_angle: Angle of the tires
        :param is_in_auto_mode: True if ego is in auto mode, false otherwise
        :param time_point: Timestamp of the ego state
        :param vehicle_parameters: Vehicle parameters
        :param tire_steering_rate: Steering rate of tires [rad/s]
        :return: The initialized EgoState
        """
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=rear_axle_pose, vehicle_parameters=vehicle_parameters
        )
        dynamic_ego_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=rear_axle_velocity_2d,
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,
            angular_velocity=angular_vel,
            angular_acceleration=angular_accel,
            tire_steering_rate=tire_steering_rate,
        )

        return cls(
            car_footprint=car_footprint,
            dynamic_car_state=dynamic_ego_state,
            tire_steering_angle=tire_steering_angle,
            time_point=time_point,
            is_in_auto_mode=is_in_auto_mode,
        )

    @classmethod
    def build_from_center(
        cls,
        center: StateSE2,
        center_velocity_2d: StateVector2D,
        center_acceleration_2d: StateVector2D,
        tire_steering_angle: float,
        time_point: TimePoint,
        vehicle_parameters: VehicleParameters,
        is_in_auto_mode: bool = True,
        angular_vel: float = 0.0,
        angular_accel: float = 0.0,
    ) -> EgoState:
        """
        Initializer using raw parameters, assumes that the reference frame is center frame
        :param center: Pose of ego center
        :param center_velocity_2d: Vectorial velocity of Ego's center
        :param center_acceleration_2d: Vectorial acceleration of Ego's center
        :param tire_steering_angle: Angle of the tires
        :param time_point: Timestamp of the ego state
        :param vehicle_parameters: Vehicle parameters
        :param is_in_auto_mode: True if ego is in auto mode, false otherwise, defaults to True
        :param angular_vel: Angular velocity of Ego, defaults to 0.0
        :param angular_accel: Angular acceleration of Ego, defaults to 0.0
        :return: The initialized EgoState
        """
        car_footprint = CarFootprint.build_from_center(center, vehicle_parameters)
        rear_axle_to_center_dist = car_footprint.rear_axle_to_center_dist
        displacement = StateVector2D(-rear_axle_to_center_dist, 0.0)
        rear_axle_velocity_2d = get_velocity_shifted(displacement, center_velocity_2d, angular_vel)
        rear_axle_acceleration_2d = get_acceleration_shifted(
            displacement, center_acceleration_2d, angular_vel, angular_accel
        )

        dynamic_ego_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=rear_axle_to_center_dist,
            rear_axle_velocity_2d=rear_axle_velocity_2d,
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,
            angular_velocity=angular_vel,
            angular_acceleration=angular_accel,
        )

        return cls(
            car_footprint=car_footprint,
            dynamic_car_state=dynamic_ego_state,
            tire_steering_angle=tire_steering_angle,
            time_point=time_point,
            is_in_auto_mode=is_in_auto_mode,
        )


class EgoStateDot(EgoState):
    """
    A class representing the dynamics of the EgoState. This class exist mostly for clarity sake.
    """

    pass
