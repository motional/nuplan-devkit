import logging
import unittest

import numpy as np

from nuplan.planning.training.data_augmentation.kinematic_agent_augmentation import KinematicAgentAugmentor
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

logger = logging.getLogger(__name__)


class TestKinematicAgentAugmentation(unittest.TestCase):
    """Test agent augmentation with kinematic constraints."""

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
                        [3.62865120e-01, 8.67895111e-02, 4.29461658e-01],
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
                    [4.1521129e-01, 1.1039978e-01, 4.1797668e-01],
                    [5.0462860e-01, 1.4907575e-01, 3.9849171e-01],
                    [6.3200253e-01, 2.0065330e-01, 3.7100676e-01],
                    [7.9846221e-01, 2.6203236e-01, 3.3552179e-01],
                    [1.0052546e00, 3.2913640e-01, 2.9203683e-01],
                    [1.2535783e00, 3.9687237e-01, 2.4055186e-01],
                    [1.5443755e00, 4.5909974e-01, 1.8106690e-01],
                    [1.8780817e00, 5.0862163e-01, 1.1358193e-01],
                    [2.2541707e00, 5.3959757e-01, 5.0773341e-02],
                    [2.6713488e00, 5.5327171e-01, 1.4758691e-02],
                    [3.1287551e00, 5.5699998e-01, 1.5426531e-03],
                    [3.6260972e00, 5.5770481e-01, 1.2917991e-03],
                ]
            )
        )

        self.uniform_aug_targets_gt = {}
        self.uniform_aug_targets_gt['trajectory'] = Trajectory(
            data=np.array(
                [
                    [0.05273135, -0.04831281, -0.08689969],
                    [0.11795828, -0.05359042, -0.07457177],
                    [0.22317114, -0.06049316, -0.05645524],
                    [0.3684539, -0.06721046, -0.03595094],
                    [0.553826, -0.07214818, -0.01731013],
                    [0.77925223, -0.0745298, -0.00381898],
                    [1.0446922, -0.07455366, 0.00363919],
                    [1.3501287, -0.07300503, 0.00650118],
                    [1.6955612, -0.07065626, 0.00709759],
                    [2.080992, -0.06789713, 0.00721934],
                    [2.5064206, -0.06473273, 0.00765666],
                    [2.9717717, -0.06097136, 0.00850872],
                ]
            )
        )

        N = 12
        dt = 0.1
        augment_prob = 1.0
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        self.gaussian_augmentor = KinematicAgentAugmentor(
            N, dt, mean, std, low, high, augment_prob, use_uniform_noise=False
        )
        self.uniform_augmentor = KinematicAgentAugmentor(
            N, dt, mean, std, low, high, augment_prob, use_uniform_noise=True
        )

    def test_gaussian_augment(self) -> None:
        """
        Test gaussian augmentation.
        """
        aug_feature, aug_targets = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] - self.aug_feature_gt['agents'].ego[0] < 1e-04).all())
        self.assertTrue(
            (aug_targets['trajectory'].data - self.gaussian_aug_targets_gt['trajectory'].data < 1e-04).all()
        )

    def test_uniform_augment(self) -> None:
        """
        Test uniform augmentation.
        """
        features_ego = self.features['agents'].ego[0].copy()
        aug_feature, aug_targets = self.uniform_augmentor.augment(self.features, self.targets)
        self.assertTrue((abs(aug_feature['agents'].ego[0] - features_ego) <= 0.1).all())
        self.assertTrue(
            (abs(aug_targets['trajectory'].data - self.uniform_aug_targets_gt['trajectory'].data) <= 0.1).all()
        )

    def test_no_augment(self) -> None:
        """
        Test no augmentation when aug_prob is set to 0.
        """
        self.gaussian_augmentor._augment_prob = 0.0
        aug_feature, aug_targets = self.gaussian_augmentor.augment(self.features, self.targets)
        self.assertTrue((aug_feature['agents'].ego[0] == self.features['agents'].ego[0]).all())
        self.assertTrue((aug_targets['trajectory'].data == self.targets['trajectory'].data).all())

    def test_input_validation(self) -> None:
        """
        Test the augmentor's validation check.
        """
        features = {'agents': None, 'test_feature': None}
        targets = {'trajectory': None, 'test_target': None}
        self.gaussian_augmentor.validate(features, targets)

        features = {'test_feature': None}
        targets = {'test_target': None}
        self.assertRaises(AssertionError, self.gaussian_augmentor.validate, features, targets)


if __name__ == '__main__':
    unittest.main()
