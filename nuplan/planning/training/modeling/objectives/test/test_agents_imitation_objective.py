import unittest
from typing import List

import numpy as np
import numpy.typing as npt
import torch

from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario
from nuplan.planning.training.modeling.objectives.agents_imitation_objective import AgentsImitationObjective
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories


class TestAgentImitationObjective(unittest.TestCase):
    """Test agent imitation objective."""

    def setUp(self) -> None:
        """Set up test case."""
        self.target_data: List[npt.NDArray[np.float32]] = [
            np.array(
                [
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                ]
            )
        ]

        self.prediction_data: List[npt.NDArray[np.float32]] = [
            np.array(
                [
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
                ]
            )
        ]

        self.objective = AgentsImitationObjective(scenario_type_loss_weighting={})

    def test_compute_loss(self) -> None:
        """
        Test loss computation
        """
        prediction = AgentsTrajectories(data=self.prediction_data)
        target = AgentsTrajectories(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='lane_following_with_lead', scenario_type='') for _ in range(2)]

        loss = self.objective.compute(
            {"agents_trajectory": prediction.to_feature_tensor()},
            {"agents_trajectory": target.to_feature_tensor()},
            scenarios,
        )
        self.assertEqual(loss, torch.tensor(0.5))

    def test_zero_loss(self) -> None:
        """
        Test perfect prediction. The loss should be zero
        """
        target = AgentsTrajectories(data=self.target_data)
        scenarios = [CachedScenario(log_name='', token='lane_following_with_lead', scenario_type='') for _ in range(2)]

        loss = self.objective.compute(
            {"agents_trajectory": target.to_feature_tensor()},
            {"agents_trajectory": target.to_feature_tensor()},
            scenarios,
        )
        self.assertEqual(loss, torch.tensor(0.0))


if __name__ == '__main__':
    unittest.main()
