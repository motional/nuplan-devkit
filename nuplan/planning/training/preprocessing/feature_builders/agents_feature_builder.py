from typing import List, Type, cast

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory_utils import (
    convert_absolute_to_relative_poses,
    convert_absolute_to_relative_velocities,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features,
    compute_yaw_rate_from_states,
    extract_and_pad_agent_poses,
    extract_and_pad_agent_sizes,
    extract_and_pad_agent_velocities,
    filter_agents,
)


def _compute_feature(
    ego_history: List[EgoState], agent_history: List[TrackedObjects], time_stamps: List[TimePoint]
) -> Agents:
    """
    Construct agent features during simulation.
    :param ego_history: ego past trajectory comprising of EgoState
    :param agent_history: agent past trajectories [num_frames, num_agents]
    :param time_stamps: the time stamps of each frame
    :return: constructed features
    """
    anchor_ego_state = ego_history[-1]
    agent_history = filter_agents(agent_history, reverse=True)

    if len(agent_history[-1].tracked_objects) == 0:
        # Return empty array when there are no agents in the scene
        agent_features: npt.NDArray[np.float32] = np.empty(
            shape=(len(agent_history), 0, Agents.agents_states_dim()), dtype=np.float32
        )
    else:
        agent_states_horizon, _ = extract_and_pad_agent_poses(agent_history, reverse=True)
        agent_sizes_horizon, _ = extract_and_pad_agent_sizes(agent_history, reverse=True)
        agent_velocities_horizon, _ = extract_and_pad_agent_velocities(agent_history, reverse=True)

        # Get all poses relative to the ego coordinate system
        agent_relative_poses = [
            convert_absolute_to_relative_poses(anchor_ego_state.rear_axle, states) for states in agent_states_horizon
        ]

        agent_relative_velocities = [
            convert_absolute_to_relative_velocities(
                StateSE2(
                    anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                    anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
                    anchor_ego_state.rear_axle.heading,
                ),
                states,
            )
            for states in agent_velocities_horizon
        ]

        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_states(agent_states_horizon, time_stamps)

        # Append the agent box pose, velocities  together
        agent_features_list = [
            np.hstack([poses, velocities, np.expand_dims(yaw_rate, axis=1), sizes])  # type: ignore
            for poses, velocities, yaw_rate, sizes in zip(
                agent_relative_poses, agent_relative_velocities, yaw_rate_horizon.transpose(), agent_sizes_horizon
            )
        ]

        agent_features = np.stack(agent_features_list)

    ego_features = build_ego_features(ego_history, reverse=True)

    return Agents(ego=[ego_features], agents=[agent_features])


class AgentsFeatureBuilder(AbstractFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "agents"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Agents  # type: ignore

    def get_features_from_scenario(self, scenario: AbstractScenario) -> Agents:
        """Inherited, see superclass."""
        # Retrieve present/past ego states and agent boxes
        anchor_ego_state = scenario.initial_ego_state

        past_ego_states = scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )
        sampled_past_ego_states = past_ego_states + [anchor_ego_state]
        time_stamps = scenario.get_past_timestamps(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        ) + [scenario.start_time]
        # Retrieve present/future agent boxes
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        # Extract and pad features
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]

        assert len(sampled_past_ego_states) == len(sampled_past_observations), (
            "Expected the trajectory length of ego and agent to be equal. "
            f"Got ego: {len(sampled_past_ego_states)} and agent: {len(sampled_past_observations)}"
        )

        assert len(sampled_past_observations) > 2, (
            "Trajectory of length of " f"{len(sampled_past_observations)} needs to be at least 3"
        )

        return _compute_feature(sampled_past_ego_states, sampled_past_observations, time_stamps)

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> Agents:
        """Inherited, see superclass."""
        history = current_input.history
        assert isinstance(
            history.observations[0], DetectionsTracks
        ), f"Expected observation of type DetectionTracks, got {type(history.observations[0])}"

        present_ego_state, present_observation = history.current_state

        past_observations = history.observations[:-1]
        past_ego_states = history.ego_states[:-1]

        assert history.sample_interval, "SimulationHistoryBuffer sample interval is None"

        indices = sample_indices_with_time_horizon(self.num_past_poses, self.past_time_horizon, history.sample_interval)

        try:
            sampled_past_observations = [
                cast(DetectionsTracks, past_observations[-idx]).tracked_objects for idx in reversed(indices)
            ]
            sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
        except IndexError:
            raise RuntimeError(
                f"SimulationHistoryBuffer duration: {history.duration} is "
                f"too short for requested past_time_horizon: {self.past_time_horizon}. "
                f"Please increase the simulation_buffer_duration in default_simulation.yaml"
            )

        sampled_past_observations = sampled_past_observations + [
            cast(DetectionsTracks, present_observation).tracked_objects
        ]
        sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]
        time_stamps = [state.time_point for state in sampled_past_ego_states]

        return _compute_feature(sampled_past_ego_states, sampled_past_observations, time_stamps)
