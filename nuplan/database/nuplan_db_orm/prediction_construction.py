from typing import Dict, List

import numpy as np

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def _waypoint_from_lidar_box(lidar_box: LidarBox) -> Waypoint:
    """
    Creates a Waypoint from a LidarBox
    :param lidar_box: the input LidarBox
    :return: the corresponding Waypoint
    """
    pose = StateSE2(lidar_box.translation[0], lidar_box.translation[1], lidar_box.yaw)
    oriented_box = OrientedBox(pose, width=lidar_box.size[0], length=lidar_box.size[1], height=lidar_box.size[2])
    velocity = StateVector2D(lidar_box.vx, lidar_box.vy)

    waypoint = Waypoint(TimePoint(lidar_box.timestamp), oriented_box, velocity)
    return waypoint


def get_waypoints_for_agent(agent_box: LidarBox, end_timestamp: int) -> List[Waypoint]:
    """
    Extracts waypoints from a LidarBox by looking into the future samples
    :param agent_box: The first LidarBox
    :param end_timestamp: The maximal timestamp, used to stop extraction
    :return: Waypoints of the agent up to end_timestamp
    """
    agent_waypoints: List[Waypoint] = []
    # We add tolerance here to make sure that also the last end_timestamp is included in the resulting waypoints
    # such that we could use later interpolation to use the exact terminal location
    tolerance_us = 60000
    while agent_box.timestamp <= end_timestamp + tolerance_us:
        agent_waypoints.append(_waypoint_from_lidar_box(agent_box))
        agent_box = agent_box.next
        if agent_box is None:
            break

    return agent_waypoints


def interpolate_waypoints(waypoints: List[Waypoint], trajectory_sampling: TrajectorySampling) -> List[Waypoint]:
    """
    Interpolates a list of waypoints given sampling time and horizon, starting at the first waypoint timestamp.
    :param waypoints: The sample waypoints
    :param trajectory_sampling: The sampling parameters
    :return: A list of interpolated waypoints
    """
    waypoint_trajectory = InterpolatedTrajectory(waypoints)

    start_time_us = waypoints[0].time_us
    end_time_us = waypoints[-1].time_us
    time_horizon_us = int(trajectory_sampling.time_horizon * 1e6)
    step_time_us = int(trajectory_sampling.step_time * 1e6)
    # we include one more step_time_us just to include also last state
    max_horizon_time = min(end_time_us, start_time_us + time_horizon_us + step_time_us)

    # Keep the current state since the predictions could have different timestamp as detections
    interpolation_times = np.arange(start_time_us, max_horizon_time, step_time_us)

    interpolated_waypoints = [
        waypoint_trajectory.get_state_at_time(TimePoint(int(interpolation_time)))
        for interpolation_time in interpolation_times
    ]

    return interpolated_waypoints


def get_interpolated_waypoints(
    lidar_pc: LidarPc, future_trajectory_sampling: TrajectorySampling
) -> Dict[str, List[Waypoint]]:
    """
    Gets the interpolated future waypoints for the agents detected in the given LidarPC. The sampling is determined
    by the horizon length and the sampling time.
    :param lidar_pc: The starting lidar pc
    :param future_trajectory_sampling: Sampling parameters for future predictions
    :return: A dict containing interpolated waypoints for each agent track_token, empty if no waypoint available
    """
    horizon_end = lidar_pc.timestamp + int(future_trajectory_sampling.time_horizon * 1e6)

    future_waypoints = {box.track_token: get_waypoints_for_agent(box, horizon_end) for box in lidar_pc.lidar_boxes}

    agents_interpolated_waypoints = {}
    for track_token, waypoints in future_waypoints.items():
        if len(waypoints) >= 2:
            interpolated = interpolate_waypoints(waypoints, future_trajectory_sampling)
            # In case the predictions have only single waypoint, that is the same as detections and for that reason
            # we ignore them
            agents_interpolated_waypoints[track_token] = interpolated if len(interpolated) > 1 else []
        else:
            agents_interpolated_waypoints[track_token] = []

    return agents_interpolated_waypoints
