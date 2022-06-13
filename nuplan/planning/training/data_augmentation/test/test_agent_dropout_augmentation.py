import unittest
from copy import deepcopy
from typing import Any, Dict

import numpy as np

from nuplan.planning.training.data_augmentation.agent_dropout_augmentation import AgentDropoutAugmentor
from nuplan.planning.training.preprocessing.features.agents import Agents


class TestAgentDropoutAugmentation(unittest.TestCase):
    """Test agent augmentation that drops out random agents from the scene."""

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)

        self.features = {}
        self.features['agents'] = Agents(
            ego=[np.random.randn(5, 3), np.random.randn(5, 3)],
            agents=[np.random.randn(5, 20, 8), np.random.randn(5, 50, 8)],
        )

        self.targets: Dict[str, Any] = {}

        augment_prob = 1.0
        self.dropout_rate = 0.5
        self.augmentor = AgentDropoutAugmentor(augment_prob, self.dropout_rate)

    def test_augment(self) -> None:
        """
        Test augmentation.
        """
        features = deepcopy(self.features)
        aug_features, _ = self.augmentor.augment(features, self.targets)
        for agents, aug_agents in zip(self.features['agents'].agents, aug_features['agents'].agents):
            self.assertLess(aug_agents.shape[1], agents.shape[1])

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.augmentor._augment_prob = 0.0
        aug_features, _ = self.augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_features['agents'].agents[0] == self.features['agents'].agents[0]).all())


if __name__ == '__main__':
    unittest.main()
