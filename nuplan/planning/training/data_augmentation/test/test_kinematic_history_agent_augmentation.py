import unittest
from typing import Any, Dict

import numpy as np

from nuplan.planning.training.data_augmentation.kinematic_history_agent_augmentation import (
    KinematicHistoryAgentAugmentor,
)
from nuplan.planning.training.preprocessing.features.agents import Agents


class TestKinematicHistoryAgentAugmentation(unittest.TestCase):
    """
    Test agent augmentation that perturbs the current ego position and generates a feasible trajectory history that
    satisfies a set of kinematic constraints.
    """

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)
        self.radius = 50

        self.features = {}
        self.features['agents'] = Agents(
            ego=[
                np.array(
                    [
                        [6.9434252e-03, -1.0949150e-03, 2.1299818e-05],
                        [4.3259640e-03, -6.9646863e-04, -9.3163371e-06],
                        [2.4353617e-03, -3.7753209e-04, 4.7789731e-06],
                        [1.1352128e-03, -1.2731040e-04, 3.8040514e-05],
                        [1.1641532e-10, 0.0000000e00, -3.0870851e-19],
                    ]
                ),
                np.array(
                    [
                        [6.9434252e-03, -1.0949150e-03, 2.1299818e-05],
                        [4.3259640e-03, -6.9646863e-04, -9.3163371e-06],
                        [2.4353617e-03, -3.7753209e-04, 4.7789731e-06],
                        [1.1352128e-03, -1.2731040e-04, 3.8040514e-05],
                        [1.1641532e-10, 0.0000000e00, -3.0870851e-19],
                    ]
                ),
            ],
            agents=[
                self.radius * np.random.rand(5, 1, 8) + self.radius / 2,
                self.radius * np.random.rand(5, 1, 8) + self.radius / 2,
            ],
        )

        self.aug_feature_gt = {}
        self.aug_feature_gt['agents'] = Agents(
            ego=[
                np.array(
                    [
                        # values from applying one instance of augmentation at given seed,
                        # should not change without change in functionality
                        [6.94342520e-03, -1.09491500e-03, 2.12998180e-05],
                        [1.20681393e-02, -1.09217957e-03, 1.04624288e-03],
                        [2.68775601e-02, -1.05475327e-03, 4.00813782e-03],
                        [5.12891984e-02, -8.97311768e-04, 8.89057227e-03],
                        [8.52192154e-02, -4.80500022e-04, 1.56771013e-02],
                    ]
                )
            ],
            agents=[self.radius * np.random.rand(5, 1, 8) + self.radius / 2],
        )

        self.targets: Dict[str, Any] = {}

        augment_prob = 1.0
        dt = 0.1
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        self.gaussian_augmentor = KinematicHistoryAgentAugmentor(
            dt, mean, std, low, high, augment_prob, use_uniform_noise=False
        )
        self.uniform_augmentor = KinematicHistoryAgentAugmentor(
            dt, mean, std, low, high, augment_prob, use_uniform_noise=True
        )

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['agents'].ego[0] - self.aug_feature_gt['agents'].ego[0]) < 0.1).all())

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        original_feature_ego = self.features['agents'].ego[1].copy()
        aug_feature, _ = self.uniform_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['agents'].ego[1] - original_feature_ego) <= 0.1).all())

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.gaussian_augmentor._augment_prob = 0.0
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] == self.features['agents'].ego[0]).all())


if __name__ == '__main__':
    unittest.main()
