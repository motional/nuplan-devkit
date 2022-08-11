import unittest
from unittest.mock import Mock, patch

from nuplan.planning.utils.color import Color, ColorType
from nuplan.planning.utils.serialization.scene import Trajectory, TrajectoryState


class TestTrajectory(unittest.TestCase):
    """
    Test scene dataclass Trajectory
    """

    def setUp(self) -> None:
        """
        Set up
        """
        self.color = Color(1, 0.5, 0, 1, ColorType.INT)
        self.states = [Mock(spec=TrajectoryState), Mock(spec=TrajectoryState)]
        self.trajectory_structure = Trajectory(color=self.color, states=self.states)

    def test_init(self) -> None:
        """
        Tests TrajectoryState initialization
        """
        # Assertions
        self.assertEqual(self.trajectory_structure.color, self.color)
        self.assertEqual(self.trajectory_structure.states, self.states)

    @patch("nuplan.planning.utils.serialization.scene.type")
    def test_serialize(self, mock_type: Mock) -> None:
        """
        Tests whether TrajectoryState is serializable
        """
        # Set up
        self.states[0].__iter__ = Mock(return_value=iter([["state_0", "value_0"]]))
        self.states[1].__iter__ = Mock(return_value=iter([["state_1", "value_1"]]))

        mock_type.side_effect = lambda x: TrajectoryState if isinstance(x, TrajectoryState) else type(x)

        # Call method
        result = dict(self.trajectory_structure)

        # Assertions
        self.assertEqual(
            result,
            {
                "color": self.color.to_list(),
                "states": [{"state_0": "value_0"}, {"state_1": "value_1"}],
            },
        )

    def test_update(self) -> None:
        """
        Tests whether Trajectory is compatible with dict.update()
        """
        # Setup
        scene = {"example": "unchanged", "color": "old_color"}

        # Call method
        scene.update(self.trajectory_structure)

        # Assertions
        self.assertEqual(
            scene,
            {
                "example": "unchanged",
                "color": self.color.to_list(),
                "states": self.states,
            },
        )


if __name__ == "__main__":
    unittest.main()
