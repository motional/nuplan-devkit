import unittest

import numpy as np
import torch

from nuplan.planning.training.modeling.objectives.dynamic_imitation_objective import DynamicImitationObjective
from nuplan.planning.training.preprocessing.features.dynamic_ego_trajectory import DynamicEgoTrajectory


class TestDynamicImitationObjective(unittest.TestCase):
    """Tests DynamicImitationObjective."""

    def setUp(self) -> None:
        """Set-up test data."""
        self.target_data = torch.Tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.prediction_data = torch.Tensor(
            [
                [1.0, 1.0, np.pi / 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.pi / 2, 1.0],
                [2.0, 1.0, np.pi / 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.pi / 2, 1.0],
                [3.0, 1.0, np.pi / 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.pi / 2, 1.0],
                [4.0, 1.0, np.pi / 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.pi / 2, 1.0],
                [5.0, 1.0, np.pi / 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.pi / 2, 1.0],
                [6.0, 1.0, np.pi / 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.pi / 2, 1.0],
            ]
        )

        self.objective = DynamicImitationObjective()

    def test_compute_loss(self) -> None:
        """
        Test loss computation
        """
        prediction = DynamicEgoTrajectory(data=self.prediction_data)
        target = DynamicEgoTrajectory(data=self.target_data)

        loss = self.objective.compute(
            {"dynamic_ego_trajectory": prediction.to_feature_tensor()},
            {"dynamic_ego_trajectory": target.to_feature_tensor()},
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(13.0)))

    def test_zero_loss(self) -> None:
        """
        Test perfect prediction. The loss should be zero
        """
        target = DynamicEgoTrajectory(data=self.target_data)

        loss = self.objective.compute(
            {"dynamic_ego_trajectory": target.to_feature_tensor()},
            {"dynamic_ego_trajectory": target.to_feature_tensor()},
        )
        self.assertEqual(loss, torch.tensor(0.0))


if __name__ == '__main__':
    unittest.main()
