import math

from nuplan.common.actor_state.agent import Agent, AgentType
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


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


def get_sample_car_footprint() -> CarFootprint:
    """
    Creates a sample CarFootprint.
    :return: A sample CarFootprint with arbitrary parameters
    """
    return CarFootprint(get_sample_oriented_box().center, get_pacifica_parameters())


def get_sample_dynamic_car_state() -> DynamicCarState:
    """
    Creates a sample DynamicCarState.
    :return: A sample DynamicCarState with arbitrary parameters
    """
    return DynamicCarState(1.44, StateVector2D(1.0, 2.0), StateVector2D(0.1, 0.2))


def get_sample_ego_state() -> EgoState:
    """
    Creates a sample EgoState.
    :return: A sample EgoState with arbitrary parameters
    """
    return EgoState(get_sample_car_footprint(), get_sample_dynamic_car_state(), 0.2, TimePoint(0))


def get_sample_agent(token: str = 'test', agent_type: AgentType = AgentType.VEHICLE) -> Agent:
    """
    Creates a sample Agent, the token and agent type can be specified for various testing purposes.
    :return: A sample Agent
    """
    return Agent(token, agent_type, get_sample_oriented_box(), velocity=StateVector2D(0.0, 0.0))
