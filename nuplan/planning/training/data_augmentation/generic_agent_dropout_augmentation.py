from typing import List, Optional, Tuple

import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import ParameterToScale, ScalingDirection
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class GenericAgentDropoutAugmentor(AbstractAugmentor):
    """Data augmentation that randomly drops out a part of agents in the scene."""

    def __init__(self, augment_prob: float, dropout_rate: float) -> None:
        """
        Initialize the augmentor.
        :param augment_prob: Probability between 0 and 1 of applying the data augmentation.
        :param dropout_rate: Rate of agents in the scenes to drop out - 0 means no dropout.
        """
        self._augment_prob = augment_prob
        self._dropout_rate = dropout_rate

    def augment(
        self, features: FeaturesType, targets: TargetsType, scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets

        for feature_name in features['generic_agents'].agents.keys():
            for batch_idx in range(len(features['generic_agents'].agents[feature_name])):
                num_agents = features['generic_agents'].agents[feature_name][batch_idx].shape[1]
                keep_mask = np.random.choice(
                    [True, False], num_agents, p=[1.0 - self._dropout_rate, self._dropout_rate]
                )
                agent_indices = np.arange(num_agents)[keep_mask]
                features['generic_agents'].agents[feature_name][batch_idx] = (
                    features['generic_agents'].agents[feature_name][batch_idx].take(agent_indices, axis=1)
                )

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
