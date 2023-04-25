import math
from typing import Optional

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


def get_sample_pose() -> StateSE2:
    """
    Creates a sample SE2 Pose.
    :return: A sample SE2 Pose with arbitrary parameters
    """
    return StateSE2(1.0, 2.0, math.pi / 2.0)


def get_sample_oriented_box() -> OrientedBox:
    """
    Creates a sample OrientedBox.
    :return: A sample OrientedBox with arbitrary parameters
    """
    return OrientedBox(get_sample_pose(), 4.0, 2.0, 1.5)


def get_sample_car_footprint(center: Optional[StateSE2] = None) -> CarFootprint:
    """
    Creates a sample CarFootprint.
    :param center: Vehicle's position. If none it uses the same position returned by get_sample_pose()
    :return: A sample CarFootprint with arbitrary parameters
    """
    if center:
        return CarFootprint.build_from_center(center=center, vehicle_parameters=get_pacifica_parameters())
    else:
        return CarFootprint.build_from_center(
            center=get_sample_oriented_box().center, vehicle_parameters=get_pacifica_parameters()
        )


def get_sample_dynamic_car_state(rear_axle_to_center_dist: float = 1.44) -> DynamicCarState:
    """
    Creates a sample DynamicCarState.
    :param rear_axle_to_center_dist: distance between rear axle and center [m]
    :return: A sample DynamicCarState with arbitrary parameters
    """
    return DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist, StateVector2D(1.0, 2.0), StateVector2D(0.1, 0.2)
    )


def get_sample_ego_state(center: Optional[StateSE2] = None, time_us: Optional[int] = 0) -> EgoState:
    """
    Creates a sample EgoState.
    :param center: Vehicle's position. If none it uses the same position returned by get_sample_pose()
    :param time_us: Time in microseconds
    :return: A sample EgoState with arbitrary parameters
    """
    return EgoState(
        car_footprint=get_sample_car_footprint(center),
        dynamic_car_state=get_sample_dynamic_car_state(),
        tire_steering_angle=0.2,
        time_point=TimePoint(time_us),
        is_in_auto_mode=False,
    )


def get_sample_agent(
    token: str = 'test',
    agent_type: TrackedObjectType = TrackedObjectType.VEHICLE,
    num_past_states: Optional[int] = 1,
    num_future_states: Optional[int] = 1,
) -> Agent:
    """
    Creates a sample Agent, the token and agent type can be specified for various testing purposes.
    :param token: The unique token to assign to the agent.
    :param agent_type: Classification of the agent.
    :param num_past_states: How many states to generate in the past trajectory. With None, that will be assigned to
    the past_trajectory otherwise the current state + num_past_states will be added.
    :param num_future_states: How many states to generate in the future trajectory. If `None` is passed, `None` will
    be assigned to the predictions; otherwise the current state + num_future_states will be added.
    :return: A sample Agent.
    """
    initial_timestamp = 10
    sample_oriented_box = get_sample_oriented_box()
    return Agent(
        agent_type,
        sample_oriented_box,
        metadata=SceneObjectMetadata(timestamp_us=initial_timestamp, track_token=token, track_id=None, token=token),
        velocity=StateVector2D(0.0, 0.0),
        predictions=[
            PredictedTrajectory(
                1.0,
                [
                    Waypoint(time_point=TimePoint(initial_timestamp + i * 5), oriented_box=sample_oriented_box)
                    for i in range(num_future_states + 1)
                ],
            )
        ]
        if num_future_states is not None
        else None,
        past_trajectory=PredictedTrajectory(
            1.0,
            [
                Waypoint(time_point=TimePoint(initial_timestamp - i * 5), oriented_box=sample_oriented_box)
                for i in reversed(range(num_past_states + 1))
            ],
        )
        if num_past_states is not None
        else None,
    )
