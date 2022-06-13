from typing import Any, Callable, Dict, List, Tuple, cast

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
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


def extract_and_pad_agent_states(
    agent_trajectories: List[TrackedObjects], state_extractor: Callable[[TrackedObjects], Any], reverse: bool
) -> Tuple[List[List[Any]], List[List[bool]]]:
    """
    Extract the agent states and pads it with the most recent available states. The order of the agents is also
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
    :param state_extractor: a function to extract a state from a SceneObject instance
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states, and an availability array indicate whether a agent's
    future state is available at each frame.
    """
    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    current_agents_state = {
        scene_object.track_token: state_extractor(scene_object)
        for scene_object in agent_trajectories[0].tracked_objects
    }

    # Sort to maintain consistency between time frames
    current_agents_state = sort_dict(current_agents_state)
    agent_states_horizon: List[List[Any]] = []
    agent_availabilities: List[List[bool]] = []

    non_availability = {agent_token: False for agent_token in current_agents_state.keys()}

    # Pad agents
    for tracked_objects in agent_trajectories:
        next_agents_states = {
            scene_object.track_token: state_extractor(scene_object) for scene_object in tracked_objects.tracked_objects
        }
        current_agents_state = {**current_agents_state, **next_agents_states}
        agent_states_horizon.append(list(current_agents_state.values()))

        next_agents_available = {scene_object.track_token: True for scene_object in tracked_objects.tracked_objects}
        current_agents_availability = {**non_availability, **next_agents_available}
        agent_availabilities.append(list(current_agents_availability.values()))

    if reverse:
        agent_states_horizon = agent_states_horizon[::-1]
        agent_availabilities = agent_availabilities[::-1]

    return agent_states_horizon, agent_availabilities


def extract_and_pad_agent_poses(
    agent_trajectories: List[TrackedObjects], reverse: bool = False
) -> Tuple[List[List[StateSE2]], List[List[bool]]]:
    """
    Extract and pad agent poses along the given trajectory. For details see extract_and_pad_agent_states.
    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of StateSE2 for all agents, and an availability array indicate whether a agent's
    future state is available at each frame.
    """
    return extract_and_pad_agent_states(
        agent_trajectories,
        lambda scene_object: StateSE2(scene_object.center.x, scene_object.center.y, scene_object.center.heading),
        reverse,
    )


def extract_and_pad_agent_sizes(
    agent_trajectories: List[TrackedObjects], reverse: bool = False
) -> Tuple[List[List[npt.NDArray[np.float32]]], List[List[bool]]]:
    """
    Extract and pad agent sizes along the given trajectory. For details see extract_and_pad_agent_states.
    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of sizes for all agents, and an availability array indicate whether a agent's
    future state is available at each frame.
    """
    return extract_and_pad_agent_states(
        agent_trajectories, lambda agent: np.array([agent.box.width, agent.box.length], np.float32), reverse
    )


def extract_and_pad_agent_velocities(
    agent_trajectories: List[TrackedObjects], reverse: bool = False
) -> Tuple[List[List[StateSE2]], List[List[bool]]]:
    """
    Extract and pad agent sizes along the given trajectory. For details see extract_and_pad_agent_states.
    :param agent_trajectories: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of velocities for all agents, and an availability array indicate whether a agent's
    future state is available at each frame.
    """
    return extract_and_pad_agent_states(
        agent_trajectories,
        lambda box: StateSE2(0, 0, 0)
        if np.isnan(box.velocity.array).any()
        else StateSE2(box.velocity.x, box.velocity.y, box.center.heading),
        reverse,
    )


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


def filter_agents(tracked_objects_history: List[TrackedObjects], reverse: bool = False) -> List[TrackedObjects]:
    """
    Filter detections to keep only Vehicles which appear in the first frame (or last frame if reverse=True)
    :param tracked_objects_history: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the last element in the list will be used as the filter
    :return: filtered agents in the same format [num_frames, num_agents]
    """
    if reverse:
        agent_tokens = [
            box.track_token
            for box in tracked_objects_history[-1].get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        ]
    else:
        agent_tokens = [
            box.track_token for box in tracked_objects_history[0].get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        ]

    filtered_agents = [
        TrackedObjects(
            [
                agent
                for agent in tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
                if agent.track_token in agent_tokens
            ]
        )
        for tracked_objects in tracked_objects_history
    ]

    return filtered_agents


def compute_yaw_rate_from_states(
    agent_states_horizon: List[List[StateSE2]], time_stamps: List[TimePoint]
) -> npt.NDArray[np.float32]:
    """
    Computes the yaw rate of all agents over the trajectory from heading
    :param agent_states_horizon: agent trajectories [num_frames, num_agents, 1]
           where each state is represented by StateSE2
    :param time_stamps: the time stamps of each frame
    :return: <np.ndarray: num_frames, num_agents, 1> where last dimension is the yaw rate
    """
    yaw: npt.NDArray[np.float32] = np.array(
        [[agent.heading for agent in frame] for frame in agent_states_horizon], dtype=np.float32
    )
    yaw_rate_horizon = approximate_derivatives(
        yaw.transpose(), np.array([stamp.time_s for stamp in time_stamps]), window_length=3
    )

    return cast(npt.NDArray[np.float32], yaw_rate_horizon)
