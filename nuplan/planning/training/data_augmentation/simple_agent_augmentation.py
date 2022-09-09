from typing import List, Optional, Tuple, cast

import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class SimpleAgentAugmentor(AbstractAugmentor):
    """Simple data augmentation that adds Gaussian noise to the ego current position with specified mean and std."""

    def __init__(
        self,
        mean: List[float],
        std: List[float],
        low: List[float],
        high: List[float],
        augment_prob: float,
        use_uniform_noise: bool = False,
    ) -> None:
        """
        Initialize the augmentor.
        :param mean: mean of 3-dimensional Gaussian noise to [x, y, yaw]
        :param std: standard deviation of 3-dimenstional Gaussian noise to [x, y, yaw]
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        self._random_offset_generator = UniformNoise(low, high) if use_uniform_noise else GaussianNoise(mean, std)
        self._augment_prob = augment_prob

    def augment(
        self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets

        for batch_idx in range(len(features['agents'].ego)):
            features['agents'].ego[batch_idx][-1] += self._random_offset_generator.sample()

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

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
