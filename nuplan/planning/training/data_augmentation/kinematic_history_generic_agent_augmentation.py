import logging
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ConstrainedNonlinearSmoother,
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

logger = logging.getLogger(__name__)


class KinematicHistoryGenericAgentAugmentor(AbstractAugmentor):
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory history that
    satisfies a set of kinematic constraints.

    This involves constrained minimization of the following objective:
    * minimize dist(perturbed_trajectory, ground_truth_trajectory)


    Simple data augmentation that adds Gaussian noise to the ego current position with specified mean and std.
    """

    def __init__(
        self,
        dt: float,
        mean: List[float],
        std: List[float],
        low: List[float],
        high: List[float],
        augment_prob: float,
        use_uniform_noise: bool = False,
    ) -> None:
        """
        Initialize the augmentor.
        :param dt: Time interval between trajectory points.
        :param mean: mean of 3-dimensional Gaussian noise to [x, y, yaw]
        :param std: standard deviation of 3-dimenstional Gaussian noise to [x, y, yaw]
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._dt = dt
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)
        self._augment_prob = augment_prob

    def safety_check(self, ego: npt.NDArray[np.float32], all_agents: List[npt.NDArray[np.float32]]) -> bool:
        """
        Check if the augmented trajectory violates any safety check (going backwards, collision with other agents).
        :param ego: Perturbed ego feature tensor to be validated.
        :param all_agents: List of agent features to validate against.
        :return: Bool reflecting feature validity.
        """
        # Check if ego goes backward after the perturbation
        if np.diff(ego, axis=0)[-1][0] < 0.0001:
            return False

        # Check if there is collision between ego and other agents
        for agents in all_agents:
            dist_to_the_closest_agent = np.min(np.linalg.norm(np.array(agents)[:, :, :2] - ego[-1, :2], axis=1))
            if dist_to_the_closest_agent < 2.5:
                return False
        return True

    def augment(
        self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets

        # Augment the history to match the distribution shift in close loop rollout
        for batch_idx in range(len(features['generic_agents'].ego)):
            trajectory_length = len(features['generic_agents'].ego[batch_idx]) - 1
            _optimizer = ConstrainedNonlinearSmoother(trajectory_length, self._dt)

            ego_trajectory: npt.NDArray[np.float32] = np.copy(features['generic_agents'].ego[batch_idx])
            ego_trajectory[-1][:3] += self._random_offset_generator.sample()
            ego_x, ego_y, ego_yaw, ego_vx, ego_vy, ego_ax, ego_ay = ego_trajectory.T
            ego_velocity = np.linalg.norm(ego_trajectory[:, 3:5], axis=1)

            # Define the 'earliest history state' as a boundary condition, and reference trajectory
            x_curr = [ego_x[0], ego_y[0], ego_yaw[0], ego_velocity[0]]
            ref_traj = ego_trajectory[:, :3]

            # Set reference and solve
            _optimizer.set_reference_trajectory(x_curr, ref_traj)

            try:
                sol = _optimizer.solve()
            except RuntimeError:
                logger.error("Smoothing failed with status %s! Use G.T. instead" % sol.stats()['return_status'])
                return features, targets

            if not sol.stats()['success']:
                logger.warning("Smoothing failed with status %s! Use G.T. instead" % sol.stats()['return_status'])
                return features, targets

            ego_perturb: npt.NDArray[np.float32] = np.vstack(
                [
                    sol.value(_optimizer.position_x),
                    sol.value(_optimizer.position_y),
                    sol.value(_optimizer.yaw),
                    sol.value(_optimizer.speed) * np.cos(sol.value(_optimizer.yaw)),
                    sol.value(_optimizer.speed) * np.sin(sol.value(_optimizer.yaw)),
                    np.concatenate((sol.value(_optimizer.accel), np.zeros(1))) * np.cos(sol.value(_optimizer.yaw)),
                    np.concatenate((sol.value(_optimizer.accel), np.zeros(1))) * np.sin(sol.value(_optimizer.yaw)),
                ]
            )
            ego_perturb = ego_perturb.T
            agents: List[npt.NDArray[np.float32]] = [
                agent_features[batch_idx] for agent_features in features['generic_agents'].agents.values()
            ]

            if self.safety_check(ego_perturb, agents):
                features["generic_agents"].ego[batch_idx] = np.float32(ego_perturb)

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['generic_agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=self._augment_prob,
            param_name=f'{self._augment_prob=}'.partition('=')[0].split('.')[1],
            scaling_direction=ScalingDirection.MAX,
        )

    @property
    def get_schedulable_attributes(self) -> List[ParameterToScale]:
        """Inherited, see superclass."""
        return cast(List[ParameterToScale], self._random_offset_generator.get_schedulable_attributes())
