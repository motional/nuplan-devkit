import unittest
from typing import Any, Dict

import numpy as np

from nuplan.planning.training.data_augmentation.kinematic_history_generic_agent_augmentation import (
    KinematicHistoryGenericAgentAugmentor,
)
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents


class TestKinematicHistoryGenericAgentAugmentation(unittest.TestCase):
    """
    Test agent augmentation that perturbs the current ego position and generates a feasible trajectory history that
    satisfies a set of kinematic constraints.
    """

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)
        self.radius = 50

        self.features = {}
        self.agent_features = ['VEHICLE', 'BICYCLE', 'PEDESTRIAN']
        self.features['generic_agents'] = GenericAgents(
            ego=[
                np.array(
                    [
                        [6.9434252e-03, -1.0949150e-03, 2.1299818e-05, 0.0, 0.0, 0.0, 0.0],
                        [4.3259640e-03, -6.9646863e-04, -9.3163371e-06, 0.0, 0.0, 0.0, 0.0],
                        [2.4353617e-03, -3.7753209e-04, 4.7789731e-06, 0.0, 0.0, 0.0, 0.0],
                        [1.1352128e-03, -1.2731040e-04, 3.8040514e-05, 0.0, 0.0, 0.0, 0.0],
                        [1.1641532e-10, 0.0000000e00, -3.0870851e-19, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                np.array(
                    [
                        [6.9434252e-03, -1.0949150e-03, 2.1299818e-05, 0.0, 0.0, 0.0, 0.0],
                        [4.3259640e-03, -6.9646863e-04, -9.3163371e-06, 0.0, 0.0, 0.0, 0.0],
                        [2.4353617e-03, -3.7753209e-04, 4.7789731e-06, 0.0, 0.0, 0.0, 0.0],
                        [1.1352128e-03, -1.2731040e-04, 3.8040514e-05, 0.0, 0.0, 0.0, 0.0],
                        [1.1641532e-10, 0.0000000e00, -3.0870851e-19, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ],
            agents={
                feature_name: [
                    self.radius * np.random.rand(5, 1, 8) + self.radius / 2,
                    self.radius * np.random.rand(5, 1, 8) + self.radius / 2,
                ]
                for feature_name in self.agent_features
            },
        )

        for sample_idx in range(len(self.features['generic_agents'].ego)):
            # derived velocities
            self.features['generic_agents'].ego[sample_idx][:-1, 3:5] = np.diff(
                self.features['generic_agents'].ego[sample_idx][:, :2], axis=0
            )
            # derived accelerations
            self.features['generic_agents'].ego[sample_idx][:-1, 5:] = np.diff(
                self.features['generic_agents'].ego[sample_idx][:, 3:5], axis=0
            )

        self.aug_feature_gt = {}
        self.aug_feature_gt['generic_agents'] = GenericAgents(
            ego=[
                np.array(
                    [
                        # values from applying one instance of augmentation at given seed,
                        # should not change without change in functionality
                        [6.94342520e-03, -1.09491500e-03, 2.12998180e-05, 0.0, 0.0, 0.0, 0.0],
                        [1.20681393e-02, -1.09217957e-03, 1.04624288e-03, 0.0, 0.0, 0.0, 0.0],
                        [2.68775601e-02, -1.05475327e-03, 4.00813782e-03, 0.0, 0.0, 0.0, 0.0],
                        [5.12891984e-02, -8.97311768e-04, 8.89057227e-03, 0.0, 0.0, 0.0, 0.0],
                        [8.52192154e-02, -4.80500022e-04, 1.56771013e-02, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ],
            agents={
                feature_name: [self.radius * np.random.rand(5, 1, 8) + self.radius / 2]
                for feature_name in self.agent_features
            },
        )

        for sample_idx in range(len(self.aug_feature_gt['generic_agents'].ego)):
            # derived velocities
            self.aug_feature_gt['generic_agents'].ego[sample_idx][:-1, 3:5] = np.diff(
                self.aug_feature_gt['generic_agents'].ego[sample_idx][:, :2], axis=0
            )
            # derived accelerations
            self.aug_feature_gt['generic_agents'].ego[sample_idx][:-1, 5:] = np.diff(
                self.aug_feature_gt['generic_agents'].ego[sample_idx][:, 3:5], axis=0
            )

        self.targets: Dict[str, Any] = {}

        augment_prob = 1.0
        dt = 0.1
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        self.gaussian_augmentor = KinematicHistoryGenericAgentAugmentor(
            dt, mean, std, low, high, augment_prob, use_uniform_noise=False
        )
        self.uniform_augmentor = KinematicHistoryGenericAgentAugmentor(
            dt, mean, std, low, high, augment_prob, use_uniform_noise=True
        )

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue(
            (
                abs(aug_feature['generic_agents'].ego[0][:, :3] - self.aug_feature_gt['generic_agents'].ego[0][:, :3])
                < 0.1
            ).all()
        )

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        original_feature_ego = self.features['generic_agents'].ego[1].copy()[:, :3]
        aug_feature, _ = self.uniform_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['generic_agents'].ego[1][:, :3] - original_feature_ego) <= 0.1).all())

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.gaussian_augmentor._augment_prob = 0.0
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['generic_agents'].ego[0] == self.features['generic_agents'].ego[0]).all())


if __name__ == '__main__':
    unittest.main()
