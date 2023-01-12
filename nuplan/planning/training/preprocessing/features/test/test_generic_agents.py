import copy
import unittest
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch

from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents


class TestGenericAgents(unittest.TestCase):
    """Test agent feature representation."""

    def setUp(self) -> None:
        """Set up test case."""
        self.agent_features = [
            'VEHICLE',
            'PEDESTRIAN',
            'BICYCLE',
            'TRAFFIC_CONE',
            'BARRIER',
            'CZONE_SIGN',
            'GENERIC_OBJECT',
        ]
        self.ego: List[npt.NDArray[np.float32]] = [np.array(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))]
        self.ego_incorrect: List[npt.NDArray[np.float32]] = [np.array([0.0, 0.0, 0.0])]

        self.agents: Dict[str, List[npt.NDArray[np.float32]]] = {
            feature_name: [
                np.array(
                    [
                        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                    ]
                )
            ]
            for feature_name in self.agent_features
        }
        self.agents_incorrect: Dict[str, List[npt.NDArray[np.float32]]] = {
            feature_name: [
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ]
            for feature_name in self.agent_features
        }

    def test_agent_feature(self) -> None:
        """
        Test the core functionality of features
        """
        feature = GenericAgents(ego=self.ego, agents=self.agents)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(GenericAgents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], np.ndarray)
        for feature_name in self.agent_features:
            self.assertIsInstance(feature.agents[feature_name][0], np.ndarray)
            self.assertIsInstance(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0), np.ndarray)
            self.assertEqual(
                feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0).shape,
                (2, feature.agents_features_dim),
            )

        feature = feature.to_feature_tensor()
        self.assertIsInstance(feature.ego[0], torch.Tensor)
        for feature_name in self.agent_features:
            self.assertIsInstance(feature.agents[feature_name][0], torch.Tensor)
            self.assertIsInstance(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0), torch.Tensor)
            self.assertEqual(
                feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0).shape,
                (2, feature.agents_features_dim),
            )

    def test_no_agents(self) -> None:
        """
        Test when there are no agents
        """
        agents: Dict[str, List[npt.NDArray[np.float32]]] = {
            feature_name: [np.empty((self.ego[0].shape[0], 0, 8), dtype=np.float32)]
            for feature_name in self.agent_features
        }

        feature = GenericAgents(ego=self.ego, agents=agents)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(GenericAgents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], np.ndarray)
        for feature_name in self.agent_features:
            self.assertIsInstance(feature.agents[feature_name][0], np.ndarray)
            self.assertIsInstance(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0), np.ndarray)
            self.assertEqual(
                feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0).shape,
                (0, feature.agents_features_dim),
            )
            self.assertEqual(feature.num_agents_in_sample(feature_name, 0), 0)

        # Test with torch tensor data type
        feature = feature.to_feature_tensor()
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(GenericAgents.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], torch.Tensor)
        for feature_name in self.agent_features:
            self.assertIsInstance(feature.agents[feature_name][0], torch.Tensor)
            self.assertIsInstance(feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0), torch.Tensor)
            self.assertEqual(
                feature.get_flatten_agents_features_by_type_in_sample(feature_name, 0).shape,
                (0, feature.agents_features_dim),
            )
            self.assertEqual(feature.num_agents_in_sample(feature_name, 0), 0)

    def test_incorrect_dimension(self) -> None:
        """
        Test when inputs dimension are incorrect
        """
        with self.assertRaises(AssertionError):
            GenericAgents(ego=self.ego, agents=self.agents_incorrect)
        with self.assertRaises(AssertionError):
            GenericAgents(ego=self.ego_incorrect, agents=self.agents)

        # Test when number of frames is inconsistent
        agents: Dict[str, List[npt.NDArray[np.float32]]] = {
            feature_name: [np.empty((self.ego[0].shape[0] + 1, 0, 8), dtype=np.float32)]
            for feature_name in self.agent_features
        }
        with self.assertRaises(AssertionError):
            GenericAgents(ego=self.ego, agents=agents)

        ego = copy.copy(self.ego)
        ego.append(np.zeros((self.ego[0].shape[0] + 1, self.ego[0].shape[1]), dtype=np.float32))

        with self.assertRaises(AssertionError):
            GenericAgents(ego=ego, agents=self.agents)

        with self.assertRaises(AssertionError):
            GenericAgents(ego=ego, agents=agents)


if __name__ == '__main__':
    unittest.main()
