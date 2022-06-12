import unittest

import numpy as np

from nuplan.planning.training.data_augmentation.gaussian_smooth_agent_augmentation import GaussianSmoothAgentAugmentor
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class TestGaussianSmoothAgentAugmentation(unittest.TestCase):
    """Test agent augmentation with gaussian smooth noise."""

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
                )
            ],
            agents=[np.random.randn(5, 1, 8)],
        )

        self.targets = {}
        self.targets['trajectory'] = Trajectory(
            data=np.array(
                [
                    [-1.2336078e-03, 2.2296980e-04, -2.0750620e-05],
                    [3.2337871e-03, 3.5673147e-04, -1.1526359e-04],
                    [2.5042057e-02, 4.6393462e-04, -4.5901173e-04],
                    [2.4698858e-01, -1.5322007e-03, -1.3717031e-03],
                    [8.2662332e-01, -7.1887751e-03, -3.9011773e-03],
                    [1.7506398e00, -1.7746322e-02, -7.2191255e-03],
                    [3.0178127e00, -3.3933811e-02, -9.0915877e-03],
                    [4.5618219e00, -5.3034388e-02, -4.8586642e-03],
                    [6.3618584e00, -6.5912366e-02, 2.6488048e-04],
                    [8.3739414e00, -6.9805034e-02, 4.0571247e-03],
                    [1.0576758e01, -4.4418037e-02, 7.4823718e-03],
                    [1.2969443e01, -1.7768066e-02, 9.7025689e-03],
                ]
            )
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
                        [-2.79207866e-01, 1.23733238e-01, 1.21955765e-01],
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

        self.aug_targets_gt = {}
        self.aug_targets_gt['trajectory'] = Trajectory(
            data=np.array(
                [
                    [-3.05143267e-01, 6.34933542e-02, 3.88459056e-02],
                    [-9.05095569e-02, 3.76481991e-02, -2.14787043e-02],
                    [1.04601096e-01, 3.52352304e-02, -2.09794331e-02],
                    [3.61774919e-01, 1.28206118e-02, -1.48434895e-02],
                    [9.21066916e-01, -8.26164336e-03, -3.80704726e-03],
                    [1.84169033e00, -1.90395200e-02, -4.84469474e-03],
                    [3.09319638e00, -3.42040445e-02, -7.79489218e-03],
                    [4.62972602e00, -5.14802886e-02, -4.72468032e-03],
                    [6.41902945e00, -6.32929286e-02, -6.67336481e-05],
                    [8.42536180e00, -6.25500571e-02, 3.91751674e-03],
                    [1.05772661e01, -4.42665087e-02, 7.14091509e-03],
                    [1.19537652e01, -2.90897778e-02, 8.73378411e-03],
                ]
            )
        )

        augment_prob = 1.0
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        sigma = 5.0
        self.augmentor = GaussianSmoothAgentAugmentor(mean, std, sigma, augment_prob)

    def test_augment(self) -> None:
        """
        Test augmentation.
        """
        aug_feature, aug_targets = self.augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] - self.aug_feature_gt['agents'].ego[0] < 1e-04).all())
        self.assertTrue((aug_targets['trajectory'].data - self.aug_targets_gt['trajectory'].data < 1e-04).all())

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.augmentor._augment_prob = 0.0
        aug_feature, aug_targets = self.augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] == self.features['agents'].ego[0]).all())
        self.assertTrue((aug_targets['trajectory'].data == self.targets['trajectory'].data).all())


if __name__ == '__main__':
    unittest.main()
