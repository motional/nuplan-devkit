import unittest
from typing import List

import numpy as np
import numpy.typing as npt
import torch

from nuplan.planning.training.preprocessing.features.dynamic_ego_feature import DynamicEgoFeature


class TestDynamicEgoFeature(unittest.TestCase):
    """Test DynamicEgoFeature representaiton."""

    def setUp(self) -> None:
        """
        Set up test ego data.
        """
        self.ego: List[npt.NDArray[np.float64]] = [
            np.array(
                (
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
                )
            )
        ]
        self.ego_incorrect: List[npt.NDArray[np.float64]] = [
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ]

    def test_agent_feature(self) -> None:
        """
        Test the core functionality of features
        """
        feature = DynamicEgoFeature(ego=self.ego)
        self.assertEqual(feature.batch_size, 1)
        self.assertEqual(DynamicEgoFeature.collate([feature, feature]).batch_size, 2)
        self.assertIsInstance(feature.ego[0], np.ndarray)
        self.assertIsInstance(feature.get_present_ego_in_sample(0), np.ndarray)
        self.assertIsInstance(feature.get_present_ego_pose_in_sample(0), np.ndarray)
        self.assertIsInstance(feature.get_present_ego_velocity_in_sample(0), np.ndarray)
        self.assertIsInstance(feature.get_present_ego_acceleration_in_sample(0), np.ndarray)
        self.assertIsInstance(feature.get_present_ego_tire_steering_angle_in_sample(0), np.ndarray)
        self.assertIsInstance(feature.get_present_ego_tire_steering_rate_in_sample(0), np.ndarray)

        feature = feature.to_feature_tensor()
        self.assertIsInstance(feature.ego[0], torch.Tensor)
        self.assertIsInstance(feature.get_present_ego_in_sample(0), torch.Tensor)
        self.assertIsInstance(feature.get_present_ego_pose_in_sample(0), torch.Tensor)
        self.assertIsInstance(feature.get_present_ego_velocity_in_sample(0), torch.Tensor)
        self.assertIsInstance(feature.get_present_ego_acceleration_in_sample(0), torch.Tensor)
        self.assertIsInstance(feature.get_present_ego_tire_steering_angle_in_sample(0), torch.Tensor)
        self.assertIsInstance(feature.get_present_ego_tire_steering_rate_in_sample(0), torch.Tensor)

    def test_incorrect_dimension(self) -> None:
        """
        Test when inputs dimension are incorrect
        """
        with self.assertRaises(AssertionError):
            DynamicEgoFeature(ego=self.ego_incorrect)


if __name__ == '__main__':
    unittest.main()
