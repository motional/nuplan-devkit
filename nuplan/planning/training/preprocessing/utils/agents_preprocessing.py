from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
from nuplan.common.utils.torch_math import approximate_derivatives_tensor, unwrap
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType
from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex, EgoFeatureIndex
from nuplan.planning.training.preprocessing.features.generic_agents import (
    GenericAgentFeatureIndex,
    GenericEgoFeatureIndex,
)
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses


class EgoInternalIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
      in the Ego Trajectory Tensors.

    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the ego x position.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the ego y position.
        :return: index
        """
        return 1

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the ego heading.
        :return: index
        """
        return 2

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the ego x velocity.
        :return: index
        """
        return 3

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the ego y velocity.
        :return: index
        """
        return 4

    @staticmethod
    def ax() -> int:
        """
        The dimension corresponding to the ego x acceleration.
        :return: index
        """
        return 5

    @staticmethod
    def ay() -> int:
        """
        The dimension corresponding to the ego y acceleration.
        :return: index
        """
        return 6

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the EgoInternal buffer.
        :return: number of features.
        """
        return 7


class AgentInternalIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
      in the tensors used to compute the final Agent Feature.


    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def track_token() -> int:
        """
        The dimension corresponding to the track_token for the agent.
        :return: index
        """
        return 0

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the x velocity of the agent.
        :return: index
        """
        return 1

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the y velocity of the agent.
        :return: index
        """
        return 2

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the heading of the agent.
        :return: index
        """
        return 3

    @staticmethod
    def width() -> int:
        """
        The dimension corresponding to the width of the agent.
        :return: index
        """
        return 4

    @staticmethod
    def length() -> int:
        """
        The dimension corresponding to the length of the agent.
        :return: index
        """
        return 5

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the x position of the agent.
        :return: index
        """
        return 6

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y position of the agent.
        :return: index
        """
        return 7

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the AgentsInternal buffer.
        :return: number of features.
        """
        return 8


def _validate_ego_feature_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be an EgoFeature.
    :param feature: The tensor to validate.
    """
    # Accept features of the shape [any, EgoFeatureIndex.dim()] or [EgoFeatureIndex.dim()]
    if len(feature.shape) == 2 and feature.shape[1] == EgoFeatureIndex.dim():
        return
    if len(feature.shape) == 1 and feature.shape[0] == EgoFeatureIndex.dim():
        return

    raise ValueError(f"Improper ego feature shape: {feature.shape}.")


def _validate_agent_feature_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be an AgentFeature.
    :param feature: The tensor to validate.
    """
    if len(feature.shape) != 3 or feature.shape[2] != AgentFeatureIndex.dim():
        raise ValueError(f"Improper agent feature shape: {feature.shape}.")


def _validate_generic_ego_feature_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be a GenericEgoFeature.
    :param feature: The tensor to validate.
    """
    # Accept features of the shape [any, GenericEgoFeatureIndex.dim()] or [GenericEgoFeatureIndex.dim()]
    if len(feature.shape) == 2 and feature.shape[1] == GenericEgoFeatureIndex.dim():
        return
    if len(feature.shape) == 1 and feature.shape[0] == GenericEgoFeatureIndex.dim():
        return

    raise ValueError(f"Improper ego feature shape: {feature.shape}.")


def _validate_generic_agent_feature_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be a GenericAgentFeature.
    :param feature: The tensor to validate.
    """
    if len(feature.shape) != 3 or feature.shape[2] != GenericAgentFeatureIndex.dim():
        raise ValueError(f"Improper agent feature shape: {feature.shape}.")


def _validate_ego_internal_shape(feature: torch.Tensor, expected_first_dim: Optional[int] = None) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be an EgoInternal.
    :param feature: The tensor to validate.
    :param expected_first_dim: If None, accept either [N, EgoInternalIndex.dim()] or [EgoInternalIndex.dim()]
                                If 1, only accept [EgoInternalIndex.dim()]
                                If 2, only accept [N, EgoInternalIndex.dim()]
    """
    # Accept features of the shape [any, EgoFeatureIndex.dim()] or [EgoFeatureIndex.dim()]
    if len(feature.shape) == 2 and feature.shape[1] == EgoInternalIndex.dim():
        if expected_first_dim is None or expected_first_dim == 2:
            return
    if len(feature.shape) == 1 and feature.shape[0] == EgoInternalIndex.dim():
        if expected_first_dim is None or expected_first_dim == 1:
            return

    raise ValueError(f"Improper ego internal shape: {feature.shape}")


def _validate_agent_internal_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be an AgentInternal.
    :param feature: the tensor to validate.
    """
    if len(feature.shape) != 2 or feature.shape[1] != AgentInternalIndex.dim():
        raise ValueError(f"Improper agent internal shape: {feature.shape}")


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
    :return: ego_features: <np.ndarray: num_frames, 3>
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


def build_ego_center_features(ego_trajectory: List[EgoState], reverse: bool = False) -> FeatureDataType:
    """
    Build agent features from the ego and agents trajectory, using center of ego OrientedBox as reference points.
    :param ego_trajectory: ego trajectory comprising of EgoState [num_frames]
    :param reverse: if True, the last element in the list will be considered as the present ego state
    :return: ego_features
             ego_features: <np.ndarray: num_frames, 3>
                         The num_frames includes both present and past/future frames.
                         The last dimension is the ego pose (x, y, heading) at time t.
    """
    if reverse:
        anchor_ego_state = ego_trajectory[-1]
    else:
        anchor_ego_state = ego_trajectory[0]

    ego_poses = [ego_state.center for ego_state in ego_trajectory]
    ego_relative_poses = convert_absolute_to_relative_poses(anchor_ego_state.center, ego_poses)

    return ego_relative_poses


def filter_agents(
    tracked_objects_history: List[TrackedObjects],
    reverse: bool = False,
    allowable_types: Optional[Set[TrackedObjectType]] = None,
) -> List[TrackedObjects]:
    """
    Filter detections to keep only agents of specified types which appear in the first frame (or last frame if reverse=True)
    :param tracked_objects_history: agent trajectories [num_frames, num_agents]
    :param reverse: if True, the last element in the list will be used as the filter
    :param allowable_types: TrackedObjectTypes to filter for (optional: defaults to VEHICLE)
    :return: filtered agents in the same format [num_frames, num_agents]
    """
    if allowable_types is None:
        allowable_types = {TrackedObjectType.VEHICLE}

    if reverse:
        agent_tokens = [
            box.track_token
            for object_type in allowable_types
            for box in tracked_objects_history[-1].get_tracked_objects_of_type(object_type)
        ]
    else:
        agent_tokens = [
            box.track_token
            for object_type in allowable_types
            for box in tracked_objects_history[0].get_tracked_objects_of_type(object_type)
        ]

    filtered_agents = [
        TrackedObjects(
            [
                agent
                for object_type in allowable_types
                for agent in tracked_objects.get_tracked_objects_of_type(object_type)
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


def convert_absolute_quantities_to_relative(
    agent_states: List[torch.Tensor], ego_state: torch.Tensor
) -> List[torch.Tensor]:
    """
    Converts the agents' poses and relative velocities from absolute to ego-relative coordinates.
    :param agent_states: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """
    _validate_ego_internal_shape(ego_state, expected_first_dim=1)

    ego_pose = torch.tensor(
        [
            float(ego_state[EgoInternalIndex.x()].item()),
            float(ego_state[EgoInternalIndex.y()].item()),
            float(ego_state[EgoInternalIndex.heading()].item()),
        ],
        dtype=torch.float64,
    )

    ego_velocity = torch.tensor(
        [
            float(ego_state[EgoInternalIndex.vx()].item()),
            float(ego_state[EgoInternalIndex.vy()].item()),
            float(ego_state[EgoInternalIndex.heading()].item()),
        ],
        dtype=torch.float64,
    )

    for agent_state in agent_states:
        _validate_agent_internal_shape(agent_state)

        agent_global_poses = agent_state[
            :, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]
        ].double()
        agent_global_velocities = agent_state[
            :, [AgentInternalIndex.vx(), AgentInternalIndex.vy(), AgentInternalIndex.heading()]
        ].double()

        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        transformed_velocities = global_state_se2_tensor_to_local(
            agent_global_velocities, ego_velocity, precision=torch.float64
        )

        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()

    return agent_states


def pad_agent_states(agent_trajectories: List[torch.Tensor], reverse: bool) -> List[torch.Tensor]:
    """
    Pads the agent states with the most recent available states. The order of the agents is also
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

    :param agent_trajectories: agent trajectories [num_frames, num_agents, AgentInternalIndex.dim()], corresponding to the AgentInternalIndex schema.
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states
    """
    for traj in agent_trajectories:
        _validate_agent_internal_shape(traj)

    track_id_idx = AgentInternalIndex.track_token()
    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    key_frame = agent_trajectories[0]

    id_row_mapping: Dict[int, int] = {}
    for idx, val in enumerate(key_frame[:, track_id_idx]):
        id_row_mapping[int(val.item())] = idx

    current_state = torch.zeros((key_frame.shape[0], key_frame.shape[1]), dtype=torch.float32)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]

        # Update current frame
        for row_idx in range(frame.shape[0]):
            mapped_row: int = id_row_mapping[int(frame[row_idx, track_id_idx].item())]
            current_state[mapped_row, :] = frame[row_idx, :]

        # Save current state
        agent_trajectories[idx] = torch.clone(current_state)

    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    return agent_trajectories


def build_ego_features_from_tensor(ego_trajectory: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """
    Build agent features from the ego states
    :param ego_trajectory: ego states at past times. Tensors complying with the EgoInternalIndex schema.
    :param reverse: if True, the last element in the list will be considered as the present ego state
    :return: Tensor complying with the EgoFeatureIndex schema.
    """
    _validate_ego_internal_shape(ego_trajectory, expected_first_dim=2)

    if reverse:
        anchor_ego_state = (
            ego_trajectory[
                ego_trajectory.shape[0] - 1, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]
            ]
            .squeeze()
            .double()
        )
    else:
        anchor_ego_state = (
            ego_trajectory[0, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
            .squeeze()
            .double()
        )

    global_ego_trajectory = ego_trajectory[
        :, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]
    ].double()
    local_ego_trajectory = global_state_se2_tensor_to_local(
        global_ego_trajectory, anchor_ego_state, precision=torch.float64
    )

    # Minor optimization. The first 3 indexes in EgoFeatureIndex and EgoInternalIndex are the same.
    # Save a tensor copy and just return local_ego_trajectory
    return local_ego_trajectory.float()


def build_generic_ego_features_from_tensor(ego_trajectory: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """
    Build generic agent features from the ego states
    :param ego_trajectory: ego states at past times. Tensors complying with the EgoInternalIndex schema.
    :param reverse: if True, the last element in the list will be considered as the present ego state
    :return: Tensor complying with the GenericEgoFeatureIndex schema.
    """
    _validate_ego_internal_shape(ego_trajectory, expected_first_dim=2)

    if reverse:
        anchor_ego_pose = (
            ego_trajectory[
                ego_trajectory.shape[0] - 1, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]
            ]
            .squeeze()
            .double()
        )
        anchor_ego_velocity = (
            ego_trajectory[
                ego_trajectory.shape[0] - 1, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.heading()]
            ]
            .squeeze()
            .double()
        )
        anchor_ego_acceleration = (
            ego_trajectory[
                ego_trajectory.shape[0] - 1, [EgoInternalIndex.ax(), EgoInternalIndex.ay(), EgoInternalIndex.heading()]
            ]
            .squeeze()
            .double()
        )
    else:
        anchor_ego_pose = (
            ego_trajectory[0, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
            .squeeze()
            .double()
        )
        anchor_ego_velocity = (
            ego_trajectory[0, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.heading()]]
            .squeeze()
            .double()
        )
        anchor_ego_acceleration = (
            ego_trajectory[0, [EgoInternalIndex.ax(), EgoInternalIndex.ay(), EgoInternalIndex.heading()]]
            .squeeze()
            .double()
        )

    global_ego_poses = ego_trajectory[
        :, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]
    ].double()
    global_ego_velocities = ego_trajectory[
        :, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.heading()]
    ].double()
    global_ego_accelerations = ego_trajectory[
        :, [EgoInternalIndex.ax(), EgoInternalIndex.ay(), EgoInternalIndex.heading()]
    ].double()

    local_ego_poses = global_state_se2_tensor_to_local(global_ego_poses, anchor_ego_pose, precision=torch.float64)
    local_ego_velocities = global_state_se2_tensor_to_local(
        global_ego_velocities, anchor_ego_velocity, precision=torch.float64
    )
    local_ego_accelerations = global_state_se2_tensor_to_local(
        global_ego_accelerations, anchor_ego_acceleration, precision=torch.float64
    )

    # Minor optimization. The indices in GenericEgoFeatureIndex and EgoInternalIndex are the same.
    local_ego_trajectory: torch.Tensor = torch.empty(
        ego_trajectory.size(), dtype=torch.float32, device=ego_trajectory.device
    )
    local_ego_trajectory[:, EgoInternalIndex.x()] = local_ego_poses[:, 0].float()
    local_ego_trajectory[:, EgoInternalIndex.y()] = local_ego_poses[:, 1].float()
    local_ego_trajectory[:, EgoInternalIndex.heading()] = local_ego_poses[:, 2].float()
    local_ego_trajectory[:, EgoInternalIndex.vx()] = local_ego_velocities[:, 0].float()
    local_ego_trajectory[:, EgoInternalIndex.vy()] = local_ego_velocities[:, 1].float()
    local_ego_trajectory[:, EgoInternalIndex.ax()] = local_ego_accelerations[:, 0].float()
    local_ego_trajectory[:, EgoInternalIndex.ay()] = local_ego_accelerations[:, 1].float()

    return local_ego_trajectory


def filter_agents_tensor(agents: List[torch.Tensor], reverse: bool = False) -> List[torch.Tensor]:
    """
    Filter detections to keep only agents which appear in the first frame (or last frame if reverse=True)
    :param agents: The past agents in the scene. A list of [num_frames] tensors, each complying with the AgentInternalIndex schema
    :param reverse: if True, the last element in the list will be used as the filter
    :return: filtered agents in the same format as the input `agents` parameter
    """
    target_tensor = agents[-1] if reverse else agents[0]
    for i in range(len(agents)):
        _validate_agent_internal_shape(agents[i])

        rows: List[torch.Tensor] = []
        for j in range(agents[i].shape[0]):
            if target_tensor.shape[0] > 0:
                agent_id: float = float(agents[i][j, int(AgentInternalIndex.track_token())].item())
                is_in_target_frame: bool = bool(
                    (agent_id == target_tensor[:, AgentInternalIndex.track_token()]).max().item()
                )
                if is_in_target_frame:
                    rows.append(agents[i][j, :].squeeze())

        if len(rows) > 0:
            agents[i] = torch.stack(rows)
        else:
            agents[i] = torch.empty((0, agents[i].shape[1]), dtype=torch.float32)

    return agents


def compute_yaw_rate_from_state_tensors(
    agent_states: List[torch.Tensor],
    time_stamps: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the yaw rate of all agents over the trajectory from heading
    :param agent_states_horizon: Agent trajectories [num_frames, num_agent, AgentsInternalBuffer.dim()]
    :param time_stamps: The time stamps of each frame.
    :return: <torch.Tensor: num_frames, num_agents> of yaw rates
    """
    # Convert time_stamps to seconds
    # Shift the min timestamp to 0 to avoid loss of precision
    if len(time_stamps.shape) != 1:
        raise ValueError(f"Unexpected timestamps shape: {time_stamps.shape}")

    time_stamps_s = (time_stamps - int(torch.min(time_stamps).item())).double() * 1e-6

    yaws: List[torch.Tensor] = []
    for agent_state in agent_states:
        _validate_agent_internal_shape(agent_state)
        yaws.append(agent_state[:, AgentInternalIndex.heading()].squeeze().double())

    # Convert to agent x frame
    yaws_tensor = torch.vstack(yaws)
    yaws_tensor = yaws_tensor.transpose(0, 1)
    # Remove [-pi, pi] yaw bounds to make the signal smooth
    yaws_tensor = unwrap(yaws_tensor, dim=-1)

    yaw_rate_horizon = approximate_derivatives_tensor(yaws_tensor, time_stamps_s, window_length=3)

    # Convert back to frame x agent
    return yaw_rate_horizon.transpose(0, 1)


def sampled_past_ego_states_to_tensor(past_ego_states: List[EgoState]) -> torch.Tensor:
    """
    Converts a list of N ego states into a N x 7 tensor. The 7 fields are as defined in `EgoInternalIndex`
    :param past_ego_states: The ego states to convert.
    :return: The converted tensor.
    """
    output = torch.zeros((len(past_ego_states), EgoInternalIndex.dim()), dtype=torch.float32)
    for i in range(0, len(past_ego_states), 1):
        output[i, EgoInternalIndex.x()] = past_ego_states[i].rear_axle.x
        output[i, EgoInternalIndex.y()] = past_ego_states[i].rear_axle.y
        output[i, EgoInternalIndex.heading()] = past_ego_states[i].rear_axle.heading
        output[i, EgoInternalIndex.vx()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.x
        output[i, EgoInternalIndex.vy()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.y
        output[i, EgoInternalIndex.ax()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.x
        output[i, EgoInternalIndex.ay()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.y

    return output


def sampled_past_timestamps_to_tensor(past_time_stamps: List[TimePoint]) -> torch.Tensor:
    """
    Converts a list of N past timestamps into a 1-d tensor of shape [N]. The field is the timestamp in uS.
    :param past_time_stamps: The time stamps to convert.
    :return: The converted tensor.
    """
    flat = [t.time_us for t in past_time_stamps]
    return torch.tensor(flat, dtype=torch.int64)


def _extract_agent_tensor(
    tracked_objects: TrackedObjects, track_token_ids: Dict[str, int], object_type: TrackedObjectType
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Extracts the relevant data from the agents present in a past detection into a tensor.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a tensor as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a tensor.
    :track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :object_type: TrackedObjectType to filter agents by.
    :return: The generated tensor and the updated track_token_ids dict.
    """
    agents = tracked_objects.get_tracked_objects_of_type(object_type)
    output = torch.zeros((len(agents), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y

    return output, track_token_ids


def sampled_tracked_objects_to_tensor_list(
    past_tracked_objects: List[TrackedObjects], object_type: TrackedObjectType = TrackedObjectType.VEHICLE
) -> List[torch.Tensor]:
    """
    Tensorizes the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each tensor as described in `_extract_agent_tensor()`.
    :param past_tracked_objects: The tracked objects to tensorize.
    :param object_type: TrackedObjectType to filter agents by.
    :return: The tensorized objects.
    """
    output: List[torch.Tensor] = []
    track_token_ids: Dict[str, int] = {}
    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids = _extract_agent_tensor(past_tracked_objects[i], track_token_ids, object_type)
        output.append(tensorized)
    return output


def pack_agents_tensor(padded_agents_tensors: List[torch.Tensor], yaw_rates: torch.Tensor) -> torch.Tensor:
    """
    Combines the local padded agents states and the computed yaw rates into the final output feature tensor.
    :param padded_agents_tensors: The padded agent states for each timestamp.
        Each tensor is of shape <num_agents, len(AgentInternalIndex)> and conforms to the AgentInternalIndex schema.
    :param yaw_rates: The computed yaw rates. The tensor is of shape <num_timestamps, agent>
    :return: The final feature, a tensor of shape [timestamp, num_agents, len(AgentsFeatureIndex)] conforming to the AgentFeatureIndex Schema
    """
    if yaw_rates.shape != (len(padded_agents_tensors), padded_agents_tensors[0].shape[0]):
        raise ValueError(f"Unexpected yaw_rates tensor shape: {yaw_rates.shape}")

    agents_tensor = torch.zeros(
        (len(padded_agents_tensors), padded_agents_tensors[0].shape[0], AgentFeatureIndex.dim())
    )

    for i in range(len(padded_agents_tensors)):
        _validate_agent_internal_shape(padded_agents_tensors[i])
        agents_tensor[i, :, AgentFeatureIndex.x()] = padded_agents_tensors[i][:, AgentInternalIndex.x()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.y()] = padded_agents_tensors[i][:, AgentInternalIndex.y()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.heading()] = padded_agents_tensors[i][
            :, AgentInternalIndex.heading()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.vx()] = padded_agents_tensors[i][:, AgentInternalIndex.vx()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.vy()] = padded_agents_tensors[i][:, AgentInternalIndex.vy()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.yaw_rate()] = yaw_rates[i, :].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.width()] = padded_agents_tensors[i][
            :, AgentInternalIndex.width()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.length()] = padded_agents_tensors[i][
            :, AgentInternalIndex.length()
        ].squeeze()

    return agents_tensor
