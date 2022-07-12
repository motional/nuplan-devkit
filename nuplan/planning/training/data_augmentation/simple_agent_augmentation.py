from typing import List, Optional, Tuple

import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import GaussianNoise
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class SimpleAgentAugmentor(AbstractAugmentor):
    """Simple data augmentation that adds Gaussian noise to the ego current position with specified mean and std."""

    def __init__(self, mean: List[float], std: List[float], augment_prob: float) -> None:
        """
        Initialize the augmentor.
        :param mean: mean of 3-dimensional Gaussian noise to [x, y, yaw]
        :param std: standard deviation of 3-dimenstional Gaussian noise to [x, y, yaw]
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        """
        self._random_offset_generator = GaussianNoise(mean, std)
        self._augment_prob = augment_prob

    def augment(
        self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets

        features['agents'].ego[0][-1] += self._random_offset_generator.sample()

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []
