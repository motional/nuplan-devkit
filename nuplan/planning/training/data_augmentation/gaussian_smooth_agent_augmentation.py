from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class GaussianSmoothAgentAugmentor(AbstractAugmentor):
    """
    Augmentor that perturbs the ego's current position and future trajectory, then applies gaussian smoothing
    to generates a smooth trajectory over the current and future trajectory.
    """

    def __init__(
        self,
        mean: List[float],
        std: List[float],
        low: List[float],
        high: List[float],
        sigma: float,
        augment_prob: float,
        use_uniform_noise: bool = False,
    ) -> None:
        """
        Initialize the augmentor class.
        :param mean: Parameter to set mean vector of the Gaussian noise on [x, y, yaw].
        :param std: Parameter to set standard deviation vector of the Gaussian noise on [x, y, yaw].
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param sigma: Parameter to control the Gaussian smooth level.
        :param augment_prob: Probability between 0 and 1 of applying the data augmentation.
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._sigma = sigma
        self._augment_prob = augment_prob
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)

    def augment(
        self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets

        ego_trajectory: npt.NDArray[np.float32] = np.concatenate(
            [features['agents'].ego[0][-1:, :], targets['trajectory'].data]
        )

        trajectory_length, trajectory_dim = ego_trajectory.shape
        ego_trajectory += np.array(
            [self._random_offset_generator.sample() for _ in range(trajectory_length)]
        ) * np.expand_dims(np.exp(-np.arange(trajectory_length)), axis=1)

        ego_x, ego_y, ego_yaw = ego_trajectory.T
        step_t = np.linspace(0, 1, len(ego_x))
        step_resample_t = np.linspace(0, 1, 100)

        ego_resample_x = np.interp(step_resample_t, step_t, ego_x)
        ego_resample_y = np.interp(step_resample_t, step_t, ego_y)
        ego_resample_yaw = np.interp(step_resample_t, step_t, ego_yaw)
        ego_perturb_x = gaussian_filter1d(ego_resample_x, self._sigma)
        ego_perturb_y = gaussian_filter1d(ego_resample_y, self._sigma)
        ego_perturb_yaw = gaussian_filter1d(ego_resample_yaw, self._sigma)

        ego_perturb_x = np.interp(step_t, step_resample_t, ego_perturb_x)
        ego_perturb_y = np.interp(step_t, step_resample_t, ego_perturb_y)
        ego_perturb_yaw = np.interp(step_t, step_resample_t, ego_perturb_yaw)
        ego_perturb: npt.NDArray[np.float32] = np.vstack((ego_perturb_x, ego_perturb_y, ego_perturb_yaw)).T

        features["agents"].ego[0][-1] = ego_perturb[0]
        targets["trajectory"].data = ego_perturb[1:]

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ['trajectory']

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
