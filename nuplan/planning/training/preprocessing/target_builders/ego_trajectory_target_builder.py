from __future__ import annotations

from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


class EgoTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Trajectory builders constructed the desired ego's trajectory from a scenario."""

    def __init__(self, future_trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes the class.
        :param future_trajectory_sampling: parameters for sampled future trajectory
        """
        self._num_future_poses = future_trajectory_sampling.num_poses
        self._time_horizon = future_trajectory_sampling.time_horizon

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "trajectory"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Trajectory  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> Trajectory:
        """Inherited, see superclass."""
        current_absolute_state = scenario.initial_ego_state
        trajectory_absolute_states = scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self._num_future_poses, time_horizon=self._time_horizon
        )

        # Get all future poses relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        if len(trajectory_relative_poses) != self._num_future_poses:
            raise RuntimeError(f'Expected {self._num_future_poses} num poses but got {len(trajectory_relative_poses)}')

        return Trajectory(data=trajectory_relative_poses)
