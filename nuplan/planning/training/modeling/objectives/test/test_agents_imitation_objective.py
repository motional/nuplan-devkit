import unittest

import numpy as np
import torch
from nuplan.planning.training.modeling.objectives.agents_imitation_objective import AgentsImitationObjective
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories


class TestAgentImitationObjective(unittest.TestCase):
    def setUp(self) -> None:

        self.target_data = [np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                                      [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])]

        self.prediction_data = [np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
                                          [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                           [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]])]

        self.objective = AgentsImitationObjective()

    def test_compute_loss(self) -> None:
        """
        Test loss computation
        """
        prediction = AgentsTrajectories(data=self.prediction_data)
        target = AgentsTrajectories(data=self.target_data)

        loss = self.objective.compute({"agents": prediction.to_feature_tensor()},
                                      {"agents": target.to_feature_tensor()})
        self.assertEqual(loss, torch.tensor(0.5))

    def test_zero_loss(self) -> None:
        """
        Test perfect prediction. The loss should be zero
        """
        target = AgentsTrajectories(data=self.target_data)

        loss = self.objective.compute({"agents": target.to_feature_tensor()}, {"agents": target.to_feature_tensor()})
        self.assertEqual(loss, torch.tensor(0.0))


if __name__ == '__main__':
    unittest.main()
