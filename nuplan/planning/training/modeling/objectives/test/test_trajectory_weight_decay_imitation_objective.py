import unittest

import numpy as np
import numpy.typing as npt
import torch

from nuplan_devkit.nuplan.planning.training.modeling.objectives.trajectory_weight_decay_imitation_objective import (
    TrajectoryWeightDecayImitationObjective,
)

from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class TestTrajectoryWeightDecayImitationObjective(unittest.TestCase):
    """Test weight decay imitation objective."""

    def setUp(self) -> None:
        """Set up test case."""
        self.target_data: npt.NDArray[np.float32] = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ]
        )

        self.prediction_data: npt.NDArray[np.float32] = np.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
            ]
        )

        self.objective = TrajectoryWeightDecayImitationObjective()

    def test_compute_loss(self) -> None:
        """
        Test loss computation
        """
        prediction = Trajectory(data=self.prediction_data)
        target = Trajectory(data=self.target_data)

        loss = self.objective.compute(
            {"trajectory": prediction.to_feature_tensor()}, {"trajectory": target.to_feature_tensor()}
        )
        torch.testing.assert_allclose(loss, torch.tensor(0.60653, dtype=torch.float64))

    def test_zero_loss(self) -> None:
        """
        Test perfect prediction. The loss should be zero
        """
        target = Trajectory(data=self.target_data)

        loss = self.objective.compute(
            {"trajectory": target.to_feature_tensor()}, {"trajectory": target.to_feature_tensor()}
        )
        self.assertEqual(loss, torch.tensor(0.0))


if __name__ == '__main__':
    unittest.main()
