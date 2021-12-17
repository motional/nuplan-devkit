from typing import Any, Callable, Dict, List, cast

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses


def sort_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sort dictionary according to the key
    :param dictionary: the dictionary to be sorted
    :return: a sorted dictionary
    """
    return {key: dictionary[key] for key in sorted(dictionary.keys())}


def extract_and_pad_agent_states(agent_trajectories: List[List[Box3D]],
                                 state_extractor: Callable[[Box3D], Any],
                                 reverse: bool) -> List[List[Any]]:
    """
    Extracts the agent states and pads it with the most recent available states. The order of the agents is also
    preserved. Note: only agents that appear in the current time step will be computed for. Agents appearing in the
    future or past will be discarded.

     t1      t2           t1      t2
    |a1,t1| |a1,t2|  pad |a1,t1| |a1,t2|
    |a2,t1| |a3,t2|  ->  |a2,t1| |a2,t1| (padded with agent 2 state at t1)
    |a3,t1| |     |      |a3,t1| |a3,t2|


    If reverse is True, the padding direction will start from the end of the trajectory towards the start

     tN-1    tN             tN-1    tN
    |a1,tN-1| |a1,tN|  pad |a1,tN-1| |a1,tN|
    |a2,tN  | |a2,tN|  <-  |a3,tN-1| |a2,tN| (padded with agent 2 state at tN)
    |a3,tN-1| |a3,tN|      |       | |a3,tN|

    :param agent_trajectories: agent trajectories [num_frames, num_agents, 1]
    :param state_extractor: a function to extract a state from a Box3D instance
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states
    """

    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    current_agents_state = {box.token: state_extractor(box)
                            for box in agent_trajectories[0]}

    # Sort to maintain consistency between time frames
    current_agents_state = sort_dict(current_agents_state)
    agent_states_horizon: List[List[Any]] = []

    # Pad agents
    for boxes in agent_trajectories:
        next_agents_states = {box.token: state_extractor(box) for box in boxes}
        current_agents_state = {**current_agents_state, **next_agents_states}
        agent_states_horizon.append(list(current_agents_state.values()))

    if reverse:
        agent_states_horizon = agent_states_horizon[::-1]

    return agent_states_horizon


def extract_and_pad_agent_poses(agent_trajectories: List[List[Box3D]], reverse: bool = False) -> List[List[StateSE2]]:
    """
    Extracts and pad agent poses along the given trajectory. For details see extract_and_pad_agent_states.

    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of StateSE2 for all agents
    """
    return extract_and_pad_agent_states(agent_trajectories,
                                        lambda box: StateSE2(box.center[0],
                                                             box.center[1],
                                                             box.yaw),
                                        reverse)


def extract_and_pad_agent_sizes(agent_trajectories: List[List[Box3D]], reverse: bool = False) \
        -> List[List[npt.NDArray[np.float32]]]:
    """
    Extracts and pad agent sizes along the given trajectory. For details see extract_and_pad_agent_states.

    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of sizes for all agents
    """
    return extract_and_pad_agent_states(agent_trajectories, lambda box: np.array(box.size[:2], np.float32), reverse)


def extract_and_pad_agent_velocities(agent_trajectories: List[List[Box3D]], reverse: bool = False) \
        -> List[List[StateSE2]]:
    """
    Extracts and pad agent sizes along the given trajectory. For details see extract_and_pad_agent_states.

    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of velocities for all agents
    """

    return extract_and_pad_agent_states(agent_trajectories,
                                        lambda box: StateSE2(0, 0, 0)
                                        if np.isnan(box.velocity).any()
                                        else StateSE2(box.velocity[0], box.velocity[1], box.yaw),
                                        reverse)


def build_ego_features(ego_trajectory: List[EgoState], reverse: bool = False) -> FeatureDataType:
    """
    Build agent features from the ego and agents trajectory

    :param ego_trajectory: ego trajectory comprising of EgoState [num_frames]
    :param reverse: if True, the last element in the list will be considered as the present ego state
    :return: Tuple[ego_features, agent_features]
             ego_features: <np.ndarray: num_frames, 3>
                         The num_frames includes both present and past/future frames.
                         The last dimension is the ego pose (x, y, heading) at time t.
    """
    if reverse:
        anchor_ego_state = ego_trajectory[-1]
    else:
        anchor_ego_state = ego_trajectory[0]

    ego_poses = [ego_state.rear_axle for ego_state in ego_trajectory]
    ego_relative_poses = convert_absolute_to_relative_poses(anchor_ego_state.rear_axle, ego_poses)
    return ego_relative_poses


def filter_agents(agent_trajectories: List[List[Box3D]], reverse: bool = False) -> List[List[Box3D]]:
    """
    Filter detections for only vehicles and agents in that appears in the first frame
    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the last element in the list will be used as the filter
    :return: filtered agents in the same format [num_frames, num_agents]
    """
    if reverse:
        agent_tokens = [box.token for box in agent_trajectories[-1]]
    else:
        agent_tokens = [box.token for box in agent_trajectories[0]]

    filtered_agents = [[box for box in boxes if box.token in agent_tokens] for boxes in agent_trajectories]

    return filtered_agents


def compute_yaw_rate_from_states(agent_states_horizon: List[List[StateSE2]], time_stamps: List[TimePoint]) \
        -> npt.NDArray[np.float32]:
    """
    Computes the yaw rate of all agents over the trajectory from heading
    :param agent_states_horizon: agent trajectories [num_frames, num_agents, 1]
           where each state is represented by StateSE2
    :param time_stamps: the time stamps of each frame
    :return: <np.ndarray: num_frames, num_agents, 1> where last dimension is the yaw rate
    """
    yaw = np.array([[agent.heading for agent in frame] for frame in agent_states_horizon], dtype=np.float32)
    yaw_rate_horizon = approximate_derivatives(yaw.transpose(), np.array([stamp.time_s for stamp in time_stamps]),
                                               window_length=3)
    return cast(npt.NDArray[np.float32], yaw_rate_horizon)
