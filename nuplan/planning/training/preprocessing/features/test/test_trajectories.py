import unittest

import torch

from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class TestTrajectories(unittest.TestCase):
    """Test trajectories target representation."""

    def setUp(self) -> None:
        """Set up test case."""
        data = torch.Tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )
        trajectory = Trajectory(data=data)
        self.trajectories = Trajectories(trajectories=[trajectory])

    def test_serialize_deserialize(self) -> None:
        """Test that serialization and deserialization work, and the resulting data matches."""
        serialized = self.trajectories.serialize()
        deserialized = Trajectories.deserialize(serialized)

        self.assertTrue(torch.allclose(self.trajectories.trajectories[0].data, deserialized.trajectories[0].data))


if __name__ == '__main__':
    unittest.main()
