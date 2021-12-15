from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.transform_state import get_front_left_corner, get_front_right_corner, \
    get_rear_left_corner, get_rear_right_corner
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from scipy.signal import savgol_filter
from shapely.geometry import Polygon


def get_rectangle_corners(
        center: StateSE2,
        half_width: float,
        half_length: float,
) -> Polygon:
    """
    Get all four corners of actor's footprint
    :param center: StateSE2 object for the center of the actor
    :param half_width: rectangle width divided by 2
    :param half_length: rectangle length divided by 2
    """
    corners = Polygon(
        [
            get_front_left_corner(center, half_length, half_width),
            get_rear_left_corner(center, half_length, half_width),
            get_rear_right_corner(center, half_length, half_width),
            get_front_right_corner(center, half_length, half_width),
        ]
    )
    return corners


def get_ego_distance_to_goal(history: SimulationHistory, goal: Point2D) -> List[float]:
    distances = []
    for hist_sample in history.data:
        ego_state = hist_sample.ego_state
        distances.append(np.hypot(ego_state.center.x - goal.x,
                                  ego_state.center.y - goal.y))

    if len(distances) == 0:
        distances = [np.nan]

    return distances


def approximate_derivatives(y: npt.NDArray[np.float32],
                            x: npt.NDArray[np.float32],
                            window_length: int = 5,
                            poly_order: int = 2,
                            deriv_order: int = 1,
                            eps_dx: float = 1e-4) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.

    :param y: The dependent variable (say of length n).
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3.
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length.
    :param deriv_order: The order of derivative to compute (default 1).
    :param eps_dx:  The maximum allowed relative difference between successive x's
        (default 1e-4).
    :return Derivatives.
    """

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = np.diff(x)  # type: ignore
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx,
    )
    return derivative


def extract_ego_time_point(history: SimulationHistory) -> npt.NDArray[int]:
    """
    Extract time point in simulation history.
    :param history: Simulation history.
    :return An array of time in micro seconds.
    """

    time_point = np.array(
        [sample.ego_state.time_point.time_us for sample in history.data]
    )
    return time_point


def extract_ego_x_position(history: SimulationHistory) -> npt.NDArray[np.float32]:
    """
    Extract x position of ego pose in simulation history.
    :param history: Simulation history.
    :return An array of ego pose in x-axis.
    """

    x = np.array([sample.ego_state.rear_axle.x for sample in history.data])
    return x


def extract_ego_y_position(history: SimulationHistory) -> npt.NDArray[np.float32]:
    """
    Extract y position of ego pose in simulation history.
    :param history: Simulation history.
    :return An array of ego pose in y-axis.
    """

    y = np.array([sample.ego_state.rear_axle.y for sample in history.data])
    return y


def extract_ego_heading(history: SimulationHistory) -> npt.NDArray[np.float32]:
    """
    Extract yaw headings of ego pose in simulation history.
    :param history: Simulation history.
    :return An array of ego pose yaw heading.
    """

    heading = np.array([sample.ego_state.rear_axle.heading for sample in history.data])
    return heading


def extract_ego_velocity(history: SimulationHistory) -> npt.NDArray[np.float32]:
    """
    Extract velocity of ego pose in simulation history.
    :param history: Simulation history.
    :return An array of ego pose velocity.
    """

    velocity = np.array([sample.ego_state.dynamic_car_state.speed for sample in history.data])
    return velocity


def extract_ego_acceleration(history: SimulationHistory) -> npt.NDArray[np.float32]:
    """
    Extract acceleration of ego pose in simulation history.
    :param history: Simulation history.
    :return An array of ego pose acceleration.
    """

    acceleration = np.array([sample.ego_state.dynamic_car_state.acceleration for sample in history.data])
    return acceleration


def extract_ego_jerk(history: SimulationHistory, accelerations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Extract jerk of ego pose in simulation history.
    :param history: Simulation history.
    :param accelerations: An array of accelerations.
    :return An array of valid ego pose jerk and timestamps.
    """

    time_points = extract_ego_time_point(history)
    jerk = approximate_derivatives(
        accelerations, time_points / 1e6
    )  # convert to seconds

    return jerk


def extract_ego_tire_steering_angle(history: SimulationHistory) -> npt.NDArray[np.float32]:
    tire_steering_angle = np.array(
        [sample.ego_state.tire_steering_angle for sample in history.data]
    )
    return tire_steering_angle


def longitudinal_projection(state_vectors: npt.NDArray[np.float32],
                            headings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Returns the signed projection of the input vectors onto the directions defined
    by the input heading angles.

    :param state_vectors: <np.float: num_vectors, 2>.  An array of input vectors.
    :param headings: <np.float: num_vectors>.  Corresponding heading angles defining
        the longitudinal direction (radians).  Need not be principal values.
    :return: <np.float: num_vectors>.  The signed magnitudes of the projections of the
        given input vectors onto the directions given by the headings.
    """

    projection: npt.NDArray[np.float32] = np.cos(headings) * state_vectors[:, 0] + \
        np.sin(headings) * state_vectors[:, 1]
    return projection


def lateral_projection(state_vectors: npt.NDArray[np.float32],
                       headings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Returns the signed projection of the input vectors onto the directions defined
    by the input heading angles plus pi/2, i.e. directions normal to the headings.

    :param state_vectors: <np.float: num_vectors, 2>.  An array of input vectors.
    :param headings: <np.float: num_vectors>.  Corresponding heading angles defining
        the longitudinal direction (radians).  Need not be principal values.
    :return: <np.float: num_vectors>.  The signed magnitudes of the projections of the
        given input vectors onto the directions normal to the headings.
    """

    projection: npt.NDArray[np.float32] = -np.sin(headings) * state_vectors[:, 0] + \
        np.cos(headings) * state_vectors[:, 1]
    return projection


def ego_delta_v_collision(ego_state: EgoState, agent: Box3D, ego_mass: float = 2000,
                          agent_mass: float = 2000) -> float:
    """
    Computes the ego delta V (loss of velocity during the collision). Delta V represents the intensity of the collision
    of the ego with other agents.

    :param ego_state: The state of ego
    :param agent: The agent ego is colliding with
    :param ego_mass: mass of ego
    :param agent_mass: mass of the agent
    :return: The delta V measure for ego
    """
    ego_mass_ratio = agent_mass / (agent_mass + ego_mass)
    agent_speed = np.linalg.norm(agent.velocity)  # type: ignore
    sum_speed_squared = ego_state.dynamic_car_state.speed ** 2 + agent_speed ** 2
    cos_rule_term = 2 * ego_state.dynamic_car_state.speed * agent_speed * np.cos(
        ego_state.rear_axle.heading - agent.yaw)
    velocity_component = float(np.sqrt(sum_speed_squared - cos_rule_term))

    delta_v_ego = ego_mass_ratio * velocity_component
    return delta_v_ego


def extract_tracks_poses(history: SimulationHistory) -> List[npt.NDArray[np.float32]]:
    """
    Extracts the pose of detected tracks to a list of N_i x 3 arrays, where N_i is the number of detections at frame i

    :param history: History from a simulation engine.
    :return: List of arrays containing poses at each timestep
    """
    track_poses: List[npt.NDArray[np.float32]] = []
    try:
        for sample in history.data:
            poses = [np.array([*box.center[:2], box.yaw]) for box in sample.observation.boxes]
            track_poses.append(np.array(poses))
    except AttributeError:
        raise AttributeError("Observations must be a list of Boxes!")

    return track_poses


def extract_tracks_speed(history: SimulationHistory) -> List[npt.NDArray[np.float32]]:
    """
    Extracts the speed of detected tracks to a list of N_i x 3 arrays, where N_i is the number of detections at frame i

    :param history: History from a simulation engine.
    :return: List of arrays containing speed at each timestep
    """
    tracks_speed: List[npt.NDArray[np.float32]] = []
    try:
        for sample in history.data:
            speeds = [np.linalg.norm(box.velocity) for box in sample.observation.boxes]  # type: ignore
            tracks_speed.append(np.array(speeds))
    except AttributeError:
        raise AttributeError("Observations must be a list of Boxes!")

    return tracks_speed
