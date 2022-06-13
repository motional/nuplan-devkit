import unittest

import torch
from torch.utils.data.dataloader import default_collate

from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class TestTrajectory(unittest.TestCase):
    """Test trajectory target representation."""

    def setUp(self) -> None:
        """Set up test case."""
        self.data = torch.Tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )
        self.batched_data = default_collate([self.data, self.data])
        self.batched_trajectory = Trajectory(data=self.batched_data)

    def test_batches(self) -> None:
        """
        Test the number of batches in trajectory
        """
        self.assertEqual(self.batched_trajectory.num_batches, 2)
        self.assertEqual(Trajectory(data=self.data).num_batches, None)

    def test_extend_trajectory(self) -> None:
        """
        Test extending trajectory by a new state
        """
        feature_builder = Trajectory(data=torch.zeros((30, 10, 3)))

        new_state = torch.zeros((30, 3)).unsqueeze(1)
        new_trajectory = Trajectory.append_to_trajectory(feature_builder, new_state)
        self.assertEqual(feature_builder.num_of_iterations + 1, new_trajectory.num_of_iterations)
        self.assertEqual(feature_builder.num_batches, 30)
        self.assertEqual(new_trajectory.num_batches, 30)

    def test_extract_trajectory(self) -> None:
        """
        Test extracting part of a trajectory
        """
        extracted = self.batched_trajectory.extract_trajectory_between(0, 4)
        self.assertEqual(extracted.data.shape, (2, 4, 3))
        self.assertAlmostEqual(extracted.data[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(extracted.data[0, -1, 0].item(), 3.0)

        state_at = self.batched_trajectory.state_at_index(3)
        state_at = state_at.unsqueeze(1)
        self.assertEqual(state_at.shape, (2, 1, 3))
        self.assertAlmostEqual(state_at[0, 0, 0], 3)


if __name__ == '__main__':
    unittest.main()
