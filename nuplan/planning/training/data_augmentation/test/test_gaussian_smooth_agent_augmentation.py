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
                        [2.67742378e-01, 5.87639301e-02, 3.05916953e-01],
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

        self.gaussian_aug_targets_gt = {}
        self.gaussian_aug_targets_gt['trajectory'] = Trajectory(
            data=np.array(
                [
                    [1.79909768e-01, 3.46292143e-02, 1.69823954e-01],
                    [1.08605770e-01, 1.97756017e-02, 4.24424040e-02],
                    [9.55989353e-02, 7.18025938e-03, 1.04373998e-02],
                    [3.29352870e-01, -9.20384090e-04, 9.19450476e-04],
                    [9.15527184e-01, -7.88396897e-03, -3.40438637e-03],
                    [1.84272996e00, -1.88512783e-02, -6.17531861e-03],
                    [3.09345437e00, -3.44966704e-02, -7.26242050e-03],
                    [4.62998953e00, -5.14234703e-02, -4.48675278e-03],
                    [6.41906077e00, -6.32653042e-02, -1.90822629e-05],
                    [8.42530270e00, -6.25461260e-02, 3.95074816e-03],
                    [1.05772538e01, -4.42685352e-02, 7.14697555e-03],
                    [1.19537668e01, -2.90942382e-02, 8.73650844e-03],
                ]
            )
        )

        self.uniform_aug_targets_gt = {}
        self.uniform_aug_targets_gt['trajectory'] = Trajectory(
            data=np.array(
                [
                    [-1.23269903e-02, 3.95750476e-03, -3.66945959e-03],
                    [5.23398530e-03, 8.76677051e-03, 4.82984929e-03],
                    [8.11338362e-02, 2.87675577e-03, -2.47428679e-03],
                    [3.41575812e-01, -2.56694967e-03, -2.36505408e-03],
                    [9.19201714e-01, -8.57337111e-03, -3.99194094e-03],
                    [1.84200781e00, -1.91452871e-02, -6.89646769e-03],
                    [3.09258798e00, -3.46222537e-02, -7.60822625e-03],
                    [4.62963856e00, -5.14547445e-02, -4.56260133e-03],
                    [6.41901491e00, -6.32771927e-02, -4.26804288e-05],
                    [8.42531047e00, -6.25500536e-02, 3.93445970e-03],
                    [1.05772518e01, -4.42706406e-02, 7.13952217e-03],
                    [1.19537637e01, -2.90951136e-02, 8.73232027e-03],
                ]
            )
        )

        augment_prob = 1.0
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        sigma = 5.0
        self.gaussian_augmentor = GaussianSmoothAgentAugmentor(
            mean, std, low, high, sigma, augment_prob, use_uniform_noise=False
        )
        self.uniform_augmentor = GaussianSmoothAgentAugmentor(
            mean, std, low, high, sigma, augment_prob, use_uniform_noise=True
        )

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, aug_targets = self.gaussian_augmentor.augment(self.features, self.targets)
        print(aug_feature, aug_targets)
        self.assertTrue((aug_feature['agents'].ego[0] - self.aug_feature_gt['agents'].ego[0] < 1e-04).all())
        self.assertTrue(
            (aug_targets['trajectory'].data - self.gaussian_aug_targets_gt['trajectory'].data < 1e-04).all()
        )

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        original_features_ego = self.features['agents'].ego[0].copy()
        aug_feature, aug_targets = self.uniform_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['agents'].ego[0] - original_features_ego) < 0.1).all())
        self.assertTrue((aug_targets['trajectory'].data - self.uniform_aug_targets_gt['trajectory'].data < 1e-04).all())

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.gaussian_augmentor._augment_prob = 0.0
        aug_feature, aug_targets = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] == self.features['agents'].ego[0]).all())
        self.assertTrue((aug_targets['trajectory'].data == self.targets['trajectory'].data).all())


if __name__ == '__main__':
    unittest.main()
