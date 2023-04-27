from typing import Any, List

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter
from shapely.geometry import Polygon

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.transform_state import (
    get_front_left_corner,
    get_front_right_corner,
    get_rear_left_corner,
    get_rear_right_corner,
)
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


def get_rectangle_corners(
    center: StateSE2,
    half_width: float,
    half_length: float,
) -> Polygon:
    """
    Get all four corners of actor's footprint
    :param center: StateSE2 object for the center of the actor
    :param half_width: rectangle width divided by 2
    :param half_length: rectangle length divided by 2.
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


def calculate_ego_progress_to_goal(ego_states: List[EgoState], goal: StateSE2) -> Any:
    """
    Progress (m) towards goal using euclidean distance assuming the goal
    does not change along the trajectory (suitable for open loop only)
    A positive number means progress to goal
    :param ego_states: A list of ego states
    :param goal: goal
    :return Progress towards goal.
    """
    if len(ego_states) > 1:
        start_distance = ego_states[0].center.distance_to(goal)
        end_distance = ego_states[-1].center.distance_to(goal)
        return start_distance - end_distance
    elif len(ego_states) == 1:
        return 0.0
    else:
        return np.nan


def get_ego_distance_to_goal(ego_states: List[EgoState], goal: StateSE2) -> List[np.float64]:
    """
    Finds the euclidean distance from the center of ego to goal
    :param ego_states: A list of ego states
    :param goal: goal
    :return A list of euclidean distance.
    """
    distances = [ego_state.center.distance_to(goal) for ego_state in ego_states]

    if len(distances) == 0:
        distances = [np.nan]

    return distances


def approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f'{poly_order} < {window_length} does not hold!')

    dx = np.diff(x)
    if not (dx > 0).all():
        raise RuntimeError('dx is not monotonically increasing!')

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y, polyorder=poly_order, window_length=window_length, deriv=deriv_order, delta=dx, axis=axis
    )
    return derivative


def extract_ego_time_point(ego_states: List[EgoState]) -> npt.NDArray[np.int32]:
    """
    Extract time point in simulation history
    :param ego_states: A list of ego stets
    :return An array of time in micro seconds.
    """
    time_point: npt.NDArray[np.int32] = np.array([ego_state.time_point.time_us for ego_state in ego_states])
    return time_point


def extract_ego_x_position(history: SimulationHistory) -> npt.NDArray[np.float32]:
    """
    Extract x position of ego pose in simulation history
    :param history: Simulation history
    :return An array of ego pose in x-axis.
    """
    x: npt.NDArray[np.float32] = np.array([sample.ego_state.rear_axle.x for sample in history.data])
    return x


def extract_ego_y_position(history: SimulationHistory) -> npt.NDArray[np.float32]:
    """
    Extract y position of ego pose in simulation history
    :param history: Simulation history
    :return An array of ego pose in y-axis.
    """
    y: npt.NDArray[np.float32] = np.array([sample.ego_state.rear_axle.y for sample in history.data])
    return y


def extract_ego_center(ego_states: List[EgoState]) -> List[Point2D]:
    """
    Extract xy position of center from a list of ego_states
    :param ego_states: list of ego states
    :return List of ego center positions.
    """
    xy_poses: List[Point2D] = [ego_state.center.point for ego_state in ego_states]
    return xy_poses


def extract_ego_center_with_heading(ego_states: List[EgoState]) -> List[StateSE2]:
    """
    Extract xy position of center and heading from a list of ego_states
    :param ego_states: list of ego states
    :return a list of StateSE2.
    """
    xy_poses_and_heading: List[StateSE2] = [ego_state.center for ego_state in ego_states]
    return xy_poses_and_heading


def extract_ego_heading(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Extract yaw headings of ego pose in simulation history
    :param ego_states: A list of ego states
    :return An array of ego pose yaw heading.
    """
    heading: npt.NDArray[np.float32] = np.array([ego_state.rear_axle.heading for ego_state in ego_states])
    return heading


def extract_ego_velocity(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Extract velocity of ego pose from list of ego states
    :param ego_states: A list of ego states
    :return An array of ego pose velocity.
    """
    velocity: npt.NDArray[np.float32] = np.array([ego_state.dynamic_car_state.speed for ego_state in ego_states])
    return velocity


def extract_ego_acceleration(
    ego_states: List[EgoState],
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float32]:
    """
    Extract acceleration of ego pose in simulation history
    :param ego_states: A list of ego states
    :param acceleration_coordinate: 'x', 'y', or 'magnitude'
    :param decimals: Decimal precision
    :return An array of ego pose acceleration.
    """
    if acceleration_coordinate == 'x':
        acceleration: npt.NDArray[np.float32] = np.asarray(
            [ego_state.dynamic_car_state.center_acceleration_2d.x for ego_state in ego_states]
        )
    elif acceleration_coordinate == 'y':
        acceleration = np.asarray([ego_state.dynamic_car_state.center_acceleration_2d.y for ego_state in ego_states])
    elif acceleration_coordinate == 'magnitude':
        acceleration = np.array([ego_state.dynamic_car_state.acceleration for ego_state in ego_states])
    else:
        raise ValueError(
            f'acceleration_coordinate option: {acceleration_coordinate} not available. '
            f'Available options are: x, y or magnitude'
        )
    acceleration = savgol_filter(
        acceleration, polyorder=poly_order, window_length=min(window_length, len(acceleration))
    )
    acceleration = np.round(acceleration, decimals=decimals)
    return acceleration


def extract_ego_jerk(
    ego_states: List[EgoState],
    acceleration_coordinate: str,
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract jerk of ego pose in simulation history
    :param ego_states: A list of ego states
    :param acceleration_coordinate: x, y or 'magnitude' in acceleration
    :param decimals: Decimal precision
    :return An array of valid ego pose jerk and timestamps.
    """
    time_points = extract_ego_time_point(ego_states)
    ego_acceleration = extract_ego_acceleration(ego_states=ego_states, acceleration_coordinate=acceleration_coordinate)

    jerk = approximate_derivatives(
        ego_acceleration,
        time_points / 1e6,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, len(ego_acceleration)),
    )  # Convert to seconds
    jerk = np.round(jerk, decimals=decimals)

    return jerk


def extract_ego_yaw_rate(
    ego_states: List[EgoState],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract ego rates
    :param ego_states: A list of ego states
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param decimals: Decimal precision
    :return An array of ego yaw rates.
    """
    ego_headings = extract_ego_heading(ego_states)
    ego_timestamps = extract_ego_time_point(ego_states)
    ego_yaw_rate = approximate_derivatives(
        phase_unwrap(ego_headings), ego_timestamps / 1e6, deriv_order=deriv_order, poly_order=poly_order
    )  # convert to seconds
    ego_yaw_rate = np.round(ego_yaw_rate, decimals=decimals)
    return ego_yaw_rate


def extract_ego_tire_steering_angle(history: SimulationHistory) -> npt.NDArray[np.float32]:
    """
    Extract ego steering angle
    :param history: Simulation history
    :return An array of ego yaw steering angle.
    """
    tire_steering_angle: npt.NDArray[np.float32] = np.array(
        [sample.ego_state.tire_steering_angle for sample in history.data]
    )
    return tire_steering_angle


def longitudinal_projection(
    state_vectors: npt.NDArray[np.float32], headings: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Returns the signed projection of the input vectors onto the directions defined
    by the input heading angles
    :param state_vectors: An array of input vectors
    :param headings: Corresponding heading angles defining
        the longitudinal direction (radians).  Need not be principal values
    :return The signed magnitudes of the projections of the
        given input vectors onto the directions given by the headings.
    """
    projection: npt.NDArray[np.float32] = (
        np.cos(headings) * state_vectors[:, 0] + np.sin(headings) * state_vectors[:, 1]
    )
    return projection


def lateral_projection(
    state_vectors: npt.NDArray[np.float32], headings: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Returns the signed projection of the input vectors onto the directions defined by the input heading angles plus pi/2, i.e. directions normal to the headings
    :param state_vectors: An array of input vectors
    :param headings: Corresponding heading angles defining the longitudinal direction (radians). Need not be principal values
    :return The signed magnitudes of the projections of the given input vectors onto the directions normal to the headings.
    """
    projection: npt.NDArray[np.float32] = (
        -np.sin(headings) * state_vectors[:, 0] + np.cos(headings) * state_vectors[:, 1]
    )
    return projection


def ego_delta_v_collision(
    ego_state: EgoState, scene_object: SceneObject, ego_mass: float = 2000, agent_mass: float = 2000
) -> float:
    """
    Computes the ego delta V (loss of velocity during the collision). Delta V represents the intensity of the collision
    of the ego with other agents.
    :param ego_state: The state of ego
    :param scene_object: The scene_object ego is colliding with
    :param ego_mass: mass of ego
    :param agent_mass: mass of the agent
    :return The delta V measure for ego
    """
    ego_mass_ratio = agent_mass / (agent_mass + ego_mass)

    scene_object_speed = scene_object.velocity.magnitude() if isinstance(scene_object, Agent) else 0

    sum_speed_squared = ego_state.dynamic_car_state.speed**2 + scene_object_speed**2
    cos_rule_term = (
        2
        * ego_state.dynamic_car_state.speed
        * scene_object_speed
        * np.cos(ego_state.rear_axle.heading - scene_object.center.heading)
    )
    velocity_component = float(np.sqrt(sum_speed_squared - cos_rule_term))

    return ego_mass_ratio * velocity_component


def extract_tracks_poses(history: SimulationHistory) -> List[npt.NDArray[np.float32]]:
    """
    Extracts the pose of detected tracks to a list of N_i x 3 arrays, where N_i is the number of detections at frame i
    :param history: History from a simulation engine.
    :return List of arrays containing poses at each timestep
    """
    track_poses: List[npt.NDArray[np.float32]] = []
    try:
        for sample in history.data:
            poses: List[npt.NDArray[np.float32]] = [
                np.array([*tracked_object.center]) for tracked_object in sample.observation.tracked_objects
            ]
            track_poses.append(np.array(poses))
    except AttributeError:
        raise AttributeError("Observations must be a list of TrackedObjects!")

    return track_poses


def extract_tracks_speed(history: SimulationHistory) -> List[npt.NDArray[np.float32]]:
    """
    Extracts the speed of detected tracks to a list of N_i x 1 arrays, where N_i is the number of detections at frame i
    :param history: History from a simulation engine
    :return List of arrays containing speed at each timestep.
    """
    tracks_speed: List[npt.NDArray[np.float32]] = []

    for sample in history.data:
        speeds = [
            np.array(tracked_object.velocity.magnitude()) if isinstance(tracked_object, Agent) else 0
            for tracked_object in sample.observation.tracked_objects
        ]
        tracks_speed.append(np.array(speeds))

    return tracks_speed


def extract_tracks_box(history: SimulationHistory) -> List[List[OrientedBox]]:
    """
    Extracts the box of detected tracks to a list of N_i list of boxes, where N_i is the number of detections at frame i
    :param history: History from a simulation engine
    :return List of lists containing tracls boxes at each timestep.
    """
    tracks_boxes: List[List[OrientedBox]] = []

    for sample in history.data:
        boxes = [tracked_object.box for tracked_object in sample.observation.tracked_objects]
        tracks_boxes.append(boxes)

    return tracks_boxes


def extract_ego_corners(ego_states: List[EgoState]) -> List[List[Point2D]]:
    """
    Extract corners of ego from a list of ego_states
    :param ego_states: List of ego states
    :return List of ego corners positions.
    """
    return [ego_state.car_footprint.all_corners() for ego_state in ego_states]


def phase_unwrap(headings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Returns an array of heading angles equal mod 2 pi to the input heading angles,
    and such that the difference between successive output angles is less than or
    equal to pi radians in absolute value
    :param headings: An array of headings (radians)
    :return The phase-unwrapped equivalent headings.
    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation of yaw to be very large.
    # We want unwrapped[j] = headings[j] - 2*pi*adjustments[j] for some integer-valued adjustments making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= headings[j+1] - headings[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (headings[j+1] - headings[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((headings[j+1] - headings[j]) / (2*pi)).
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[1:] = np.cumsum(np.round(np.diff(headings) / two_pi))
    unwrapped = headings - two_pi * adjustments
    return unwrapped
