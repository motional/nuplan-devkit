from typing import List, Type

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.features.dynamic_ego_feature import DynamicEgoFeature
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import build_ego_features


def _compute_feature(ego_history: List[EgoState]) -> DynamicEgoFeature:
    """
    Comptues DynamicEgoFeature from history of EgoStates.

    :param ego_history: ego past trajectory comprising of EgoState
    :param agent_history: agent past trajectories [num_frames, num_agents]
    :param time_stamps: the time stamps of each frame
    :return: constructed features
    """
    ego_data: npt.NDArray[np.float32] = np.zeros(
        (len(ego_history), DynamicEgoFeature.ego_state_dim()), dtype=np.float32
    )

    # Compute relative poses
    ego_relative_poses = build_ego_features(ego_history)
    ego_data[:, :3] = ego_relative_poses

    # Retrieve velocities
    longitudinal_velocities = [ego.dynamic_car_state.rear_axle_velocity_2d.x for ego in ego_history]
    lateral_velocities = [ego.dynamic_car_state.rear_axle_velocity_2d.y for ego in ego_history]
    angular_velocities = [ego.dynamic_car_state.angular_velocity for ego in ego_history]
    ego_data[:, 3] = longitudinal_velocities
    ego_data[:, 4] = lateral_velocities
    ego_data[:, 5] = angular_velocities

    # Retrieve accelerations
    longitudinal_accelerations = [ego.dynamic_car_state.rear_axle_acceleration_2d.x for ego in ego_history]
    lateral_accelerations = [ego.dynamic_car_state.rear_axle_acceleration_2d.y for ego in ego_history]
    angular_accelerations = [ego.dynamic_car_state.angular_acceleration for ego in ego_history]
    ego_data[:, 6] = longitudinal_accelerations
    ego_data[:, 7] = lateral_accelerations
    ego_data[:, 8] = angular_accelerations

    # Retrieve steering angles and rates
    steering_angles = [ego.tire_steering_angle for ego in ego_history]
    steering_rates = [ego.dynamic_car_state.tire_steering_rate for ego in ego_history]
    ego_data[:, 9] = steering_angles
    ego_data[:, 10] = steering_rates

    return DynamicEgoFeature(ego=[ego_data])


class DynamicEgoFeatureBuilder(AbstractFeatureBuilder):
    """
    Builder for constructing dynamic ego features during training and simulation.
    """

    def __init__(self, trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes DynamicEgoFeatureBuilder.

        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "dynamic_ego"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return DynamicEgoFeature  # type: ignore

    def get_features_from_scenario(self, scenario: AbstractScenario) -> DynamicEgoFeature:
        """Inherited, see superclass."""
        # Retrieve present/past ego states
        anchor_ego_state = scenario.initial_ego_state

        past_ego_states = scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )
        sampled_past_ego_states = past_ego_states + [anchor_ego_state]

        # Compute and return feature
        return _compute_feature(sampled_past_ego_states)

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> DynamicEgoFeature:
        """Inherited, see superclass."""
        # Extract present and past ego states from simulation history
        history = current_input.history
        assert isinstance(
            history.observations[0], DetectionsTracks
        ), f"Expected observation of type DetectionTracks, got {type(history.observations[0])}"

        interval_time = history.ego_states[1].time_point.time_s - history.ego_states[0].time_point.time_s
        indices = sample_indices_with_time_horizon(self.num_past_poses, self.past_time_horizon, interval_time)

        present_ego_state, present_observation = history.current_state

        past_ego_states = history.ego_states[:-1]
        sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
        sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]

        # Compute and return feature
        return _compute_feature(sampled_past_ego_states)
