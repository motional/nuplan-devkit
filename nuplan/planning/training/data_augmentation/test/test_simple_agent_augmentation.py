import unittest
from typing import Any, Dict

import numpy as np

from nuplan.planning.training.data_augmentation.simple_agent_augmentation import SimpleAgentAugmentor
from nuplan.planning.training.preprocessing.features.agents import Agents


class TestSimpleAgentAugmentation(unittest.TestCase):
    """Test agent augmentation that simply adds noise to the current ego position."""

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)

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
                np.random.randn(5, 1, 8),
                np.random.randn(5, 1, 8),
            ],
        )

        self.aug_feature_gt = {}
        self.aug_feature_gt['agents'] = Agents(
            ego=[
                np.array(
                    [
                        [6.94342520e-03, -1.09491500e-03, 2.12998180e-05],
                        [4.32596400e-03, -6.96468630e-04, -9.31633710e-06],
                        [2.43536170e-03, -3.77532090e-04, 4.77897310e-06],
                        [1.13521280e-03, -1.27310400e-04, 3.80405140e-05],
                        [3.62865111e-01, 8.67895137e-02, 4.29461646e-01],
                    ]
                )
            ],
            agents=[
                np.array(
                    [
                        [
                            [
                                -5.27899086e-04,
                                -2.74901425e-01,
                                -1.39285562e-01,
                                1.98468616e00,
                                2.82109326e-01,
                                7.60808658e-01,
                                3.00981606e-01,
                                5.40297269e-01,
                            ]
                        ],
                        [
                            [
                                3.73497287e-01,
                                3.77813394e-01,
                                -9.02131926e-02,
                                -2.30594327e00,
                                1.14276002e00,
                                -1.53565429e00,
                                -8.63752018e-01,
                                1.01654494e00,
                            ]
                        ],
                        [
                            [
                                1.03396388e00,
                                -8.24492228e-01,
                                1.89048564e-02,
                                -3.83343556e-01,
                                -3.04185475e-01,
                                9.97291506e-01,
                                -1.27273841e-01,
                                -1.47588590e00,
                            ]
                        ],
                        [
                            [
                                -1.94090633e00,
                                8.33648924e-01,
                                -5.67217888e-01,
                                1.17448696e00,
                                3.19068832e-01,
                                1.90870428e-01,
                                3.69270181e-01,
                                -1.01147863e-01,
                            ]
                        ],
                        [
                            [
                                -9.41809489e-01,
                                -1.40414171e00,
                                2.08064701e00,
                                -1.20316234e-01,
                                7.59791879e-01,
                                1.82743214e00,
                                -6.60727087e-01,
                                -8.07806261e-01,
                            ]
                        ],
                    ]
                )
            ],
        )

        self.targets: Dict[str, Any] = {}

        augment_prob = 1.0
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        self.gaussian_augmentor = SimpleAgentAugmentor(mean, std, low, high, augment_prob, use_uniform_noise=False)
        self.uniform_augmentor = SimpleAgentAugmentor(mean, std, low, high, augment_prob, use_uniform_noise=True)

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, _ = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] - self.aug_feature_gt['agents'].ego[0] < 1e-04).all())

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        original_feature_ego = self.features['agents'].ego[1].copy()
        aug_feature, _ = self.uniform_augmentor.augment(self.features, self.targets)
        print(f'{original_feature_ego}, \n {aug_feature}')
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
