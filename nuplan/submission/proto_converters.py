from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.submission import challenge_pb2 as chpb


def proto_vector_2d_from_vector_2d(vector: StateVector2D) -> chpb.StateVector2D:
    """
    Serializes StateVector2D to a StateVector2D message
    :param vector: The StateVector2D object
    :return: The corresponding StateVector2D message
    """
    return chpb.StateVector2D(x=vector.x, y=vector.y)


def vector_2d_from_proto_vector_2d(vector: chpb.StateVector2D) -> StateVector2D:
    """
    Deserializes StateVector2D message to a StateVector2D object
    :param vector: The proto StateVector2D message
    :return: The corresponding StateVector2D object
    """
    return StateVector2D(x=vector.x, y=vector.y)


def proto_se2_from_se2(se2: StateSE2) -> chpb.StateSE2:
    """
    Serializes StateSE2 to a StateSE2 message
    :param se2: The StateSE2 object
    :return: The corresponding StateSE2 message
    """
    return chpb.StateSE2(x=se2.x, y=se2.y, heading=se2.heading)


def se2_from_proto_se2(se2: chpb.StateSE2) -> StateSE2:
    """
    Deserializes StateSE2 message to a StateSE2 object
    :param se2: The proto StateSE2 message
    :return: The corresponding StateSE2 object
    """
    return StateSE2(x=se2.x, y=se2.y, heading=se2.heading)


def proto_ego_state_from_ego_state(ego_state: EgoState) -> chpb.EgoState:
    """
    Serializes EgoState to a EgoState message
    :param ego_state: The EgoState object
    :return: The corresponding EgoState message
    """
    return chpb.EgoState(
        rear_axle_pose=proto_se2_from_se2(ego_state.rear_axle),
        rear_axle_velocity_2d=proto_vector_2d_from_vector_2d(ego_state.dynamic_car_state.rear_axle_velocity_2d),
        rear_axle_acceleration_2d=proto_vector_2d_from_vector_2d(ego_state.dynamic_car_state.rear_axle_acceleration_2d),
        tire_steering_angle=ego_state.tire_steering_angle,
        time_us=ego_state.time_us,
        angular_vel=ego_state.dynamic_car_state.angular_velocity,
        angular_accel=ego_state.dynamic_car_state.angular_acceleration,
    )


def ego_state_from_proto_ego_state(ego_state: chpb.EgoState) -> EgoState:
    """
    Deserializes EgoState message to a EgoState object
    :param ego_state: The proto EgoState message
    :return: The corresponding EgoState object
    """
    vehicle_parameters = get_pacifica_parameters()
    return EgoState.build_from_rear_axle(
        rear_axle_pose=se2_from_proto_se2(ego_state.rear_axle_pose),
        rear_axle_velocity_2d=vector_2d_from_proto_vector_2d(ego_state.rear_axle_velocity_2d),
        rear_axle_acceleration_2d=vector_2d_from_proto_vector_2d(ego_state.rear_axle_acceleration_2d),
        tire_steering_angle=ego_state.tire_steering_angle,
        time_point=TimePoint(ego_state.time_us),
        angular_vel=ego_state.angular_vel,
        angular_accel=ego_state.angular_accel,
        vehicle_parameters=vehicle_parameters,
    )


def proto_traj_from_inter_traj(trajectory: AbstractTrajectory) -> chpb.Trajectory:
    """
    Serializes AbstractTrajectory to a Trajectory message
    :param trajectory: The AbstractTrajectory object
    :return: The corresponding Trajectory message
    """
    return chpb.Trajectory(
        ego_states=[proto_ego_state_from_ego_state(state) for state in trajectory.get_sampled_trajectory()]
    )


def interp_traj_from_proto_traj(trajectory: chpb.Trajectory) -> InterpolatedTrajectory:
    """
    Deserializes Trajectory message to a InterpolatedTrajectory object
    :param trajectory: The proto Trajectory message
    :return: The corresponding InterpolatedTrajectory object
    """
    return InterpolatedTrajectory([ego_state_from_proto_ego_state(state) for state in trajectory.ego_states])


def proto_tl_status_type_from_tl_status_type(tl_status_type: TrafficLightStatusType) -> chpb.TrafficLightStatusType:
    """
    Serializes TrafficLightStatusType to a TrafficLightStatusType message
    :param tl_status_type: The TrafficLightStatusType object
    :return: The corresponding TrafficLightStatusType message
    """
    return chpb.TrafficLightStatusType(status_name=tl_status_type.serialize())


def tl_status_type_from_proto_tl_status_type(tl_status_type: chpb.TrafficLightStatusType) -> TrafficLightStatusType:
    """
    Deserializes TrafficLightStatusType message to a TrafficLightStatusType object
    :param tl_status_type: The proto TrafficLightStatusType message
    :return: The corresponding TrafficLightStatusType object
    """
    return TrafficLightStatusType.deserialize(tl_status_type.status_name)


def proto_tl_status_data_from_tl_status_data(tl_status_data: TrafficLightStatusData) -> chpb.TrafficLightStatusData:
    """
    Serializes TrafficLightStatusData to a TrafficLightStatusData message
    :param tl_status_data: The TrafficLightStatusData object
    :return: The corresponding TrafficLightStatusData message
    """
    return chpb.TrafficLightStatusData(
        status=proto_tl_status_type_from_tl_status_type(tl_status_data.status),
        lane_connector_id=tl_status_data.lane_connector_id,
        timestamp=tl_status_data.timestamp,
    )


def tl_status_data_from_proto_tl_status_data(tl_status_data: chpb.TrafficLightStatusData) -> TrafficLightStatusData:
    """
    Deserializes TrafficLightStatusType message to a TrafficLightStatusType object
    :param tl_status_data: The proto TrafficLightStatusType message
    :return: The corresponding TrafficLightStatusType object
    """
    return TrafficLightStatusData(
        status=tl_status_type_from_proto_tl_status_type(tl_status_data.status),
        lane_connector_id=tl_status_data.lane_connector_id,
        timestamp=tl_status_data.timestamp,
    )
