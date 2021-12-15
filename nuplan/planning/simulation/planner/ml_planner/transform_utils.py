from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.transform_state import convert_relative_to_absolute_poses


def _state_se2_to_ego_state(state: StateSE2, timestamp: float) -> EgoState:
    """
    Convert StateSE2 to EgoState given a timestamp.

    :param state: input SE2 state
    :param timestamp: [s] timestamp of state
    :return: output agent state
    """
    return EgoState.from_raw_params(state,
                                    velocity_2d=StateVector2D(0.0, 0.0),
                                    acceleration_2d=StateVector2D(0.0, 0.0),
                                    tire_steering_angle=0.0,
                                    time_point=TimePoint(int(timestamp * 1e6)))


def _get_fixed_timesteps(state: EgoState, future_horizon: float, step_interval: float) -> List[float]:
    """
    Get a fixed array of timesteps starting from a state's time.

    :param state: input state
    :param future_horizon: [s] future time horizon
    :param step_interval: [s] interval between steps in the array
    :return: constructed timestep list
    """
    timesteps = np.arange(0.0, future_horizon, step_interval) + step_interval
    timesteps += state.time_point.time_s

    return list(timesteps.tolist())


def _get_absolute_agent_states_from_numpy_poses(
    poses: npt.NDArray[np.float32],
    ego_state: EgoState,
    timesteps: List[float],
) -> List[EgoState]:
    """
    Converts an array of relative numpy poses to a list of absolute EgoState objects.

    :param poses: input relative poses
    :param ego_state: ego state
    :param timesteps: timestamps corresponding to each state
    :return: list of agent states
    """
    relative_states = [StateSE2.deserialize(pose) for pose in poses]
    absolute_states = convert_relative_to_absolute_poses(ego_state.rear_axle, relative_states)
    agent_states = [_state_se2_to_ego_state(state, timestep) for state, timestep in zip(absolute_states, timesteps)]

    return agent_states


def transform_predictions_to_states(
    predicted_poses: npt.NDArray[np.float32],
    ego_state: EgoState,
    future_horizon: float,
    step_interval: float,
    include_ego_state: bool = True,
) -> List[EgoState]:
    """
    Transform an array of pose predictions to a list of EgoState.

    :param predicted_poses: input relative poses
    :param ego_state: ego state
    :param future_horizon: [s] future time horizon
    :param step_interval: [s] interval between steps in the array
    :param include_ego_state: whether to include the current ego state as the initial state
    :return: transformed absolute states
    """
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    states = _get_absolute_agent_states_from_numpy_poses(predicted_poses, ego_state, timesteps)

    if include_ego_state:
        states.insert(0, ego_state)

    return states
