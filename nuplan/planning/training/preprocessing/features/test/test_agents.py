import unittest

import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.agents import AgentsFeature


class TestAgentsFeature(unittest.TestCase):
    def setUp(self) -> None:
        self.ego = [np.array(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))]
        self.ego_incorrect = [np.array([0.0, 0.0, 0.0])]

        self.agents = [np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                                 [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])]
        self.agents_incorrect = [np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])]

    def test_agent_feature(self) -> None:
        """
        Test the core functionality of features
        """
        feature = AgentsFeature(ego=self.ego, agents=self.agents)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(AgentsFeature.collate([feature, feature]).batch_size, 2)
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
            AgentsFeature(ego=self.ego, agents=self.agents_incorrect)
        with self.assertRaises(AssertionError):
            AgentsFeature(ego=self.ego_incorrect, agents=self.agents)


if __name__ == '__main__':
    unittest.main()
