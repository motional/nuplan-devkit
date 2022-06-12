import unittest
from typing import List

import numpy as np
import numpy.typing as npt
import torch

from nuplan.planning.training.preprocessing.features.agents import Agents


class TestAgents(unittest.TestCase):
    """Test agent feature representation."""

    def setUp(self) -> None:
        """Set up test case."""
        self.ego: List[npt.NDArray[np.float32]] = [np.array(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))]
        self.ego_incorrect: List[npt.NDArray[np.float32]] = [np.array([0.0, 0.0, 0.0])]

        self.agents: List[npt.NDArray[np.float32]] = [
            np.array(
                [
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                ]
            )
        ]
        self.agents_incorrect: List[npt.NDArray[np.float32]] = [
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            )
        ]

    def test_agent_feature(self) -> None:
        """
        Test the core functionality of features
        """
        feature = Agents(ego=self.ego, agents=self.agents)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(Agents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], np.ndarray)
        self.assertIsInstance(feature.agents[0], np.ndarray)
        self.assertIsInstance(feature.get_flatten_agents_features_in_sample(0), np.ndarray)
        self.assertEqual(feature.get_flatten_agents_features_in_sample(0).shape, (2, feature.agents_features_dim))

        feature = feature.to_feature_tensor()
        self.assertIsInstance(feature.get_flatten_agents_features_in_sample(0), torch.Tensor)
        self.assertEqual(feature.get_flatten_agents_features_in_sample(0).shape, (2, feature.agents_features_dim))
        self.assertIsInstance(feature.ego[0], torch.Tensor)
        self.assertIsInstance(feature.agents[0], torch.Tensor)

    def test_incorrect_dimension(self) -> None:
        """
        Test when inputs dimension are incorrect
        """
        with self.assertRaises(AssertionError):
            Agents(ego=self.ego, agents=self.agents_incorrect)
        with self.assertRaises(AssertionError):
            Agents(ego=self.ego_incorrect, agents=self.agents)


if __name__ == '__main__':
    unittest.main()
