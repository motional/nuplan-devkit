import unittest

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.utils.serialization.scene import TrajectoryState


class TestTrajectoryState(unittest.TestCase):
    """
    Test scene dataclass TrajectoryState
    """

    def setUp(self) -> None:
        """
        Set up
        """
        self.pose_x = 1.12
        self.pose_y = 2.11
        self.pose_heading = 0.29
        self.pose = StateSE2(self.pose_x, self.pose_y, self.pose_heading)
        self.speed = 1.23
        self.velocity_2d = [0.12, 0.54]
        self.lateral = [0.0, 0.0]
        self.acceleration = [0.32, 0.43]

        self.trajectory_state = TrajectoryState(
            pose=self.pose,
            speed=self.speed,
            velocity_2d=self.velocity_2d,
            lateral=self.lateral,
            acceleration=self.acceleration,
        )

    def test_init(self) -> None:
        """
        Tests TrajectoryState initialization
        """
        # Assertions
        self.assertEqual(self.trajectory_state.pose, self.pose)
        self.assertEqual(self.trajectory_state.speed, self.speed)
        self.assertEqual(self.trajectory_state.velocity_2d, self.velocity_2d)
        self.assertEqual(self.trajectory_state.lateral, self.lateral)
        self.assertEqual(self.trajectory_state.acceleration, self.acceleration)

        # Uninitialized optional attributes
        self.assertIsNone(self.trajectory_state.tire_steering_angle)

    def test_serialize(self) -> None:
        """
        Tests whether TrajectoryState is serializable
        """
        # Call method
        result = dict(self.trajectory_state)

        # Asssertions
        self.assertEqual(
            result,
            {
                "pose": [self.pose_x, self.pose_y, self.pose_heading],
                "speed": self.speed,
                "velocity_2d": self.velocity_2d,
                "lateral": self.lateral,
                "acceleration": self.acceleration,
            },
        )
        self.assertFalse("tire_steering_angle" in result.keys())

    def test_update(self) -> None:
        """
        Tests whether TrajectoryState is compatible with dict.update()
        """
        # Setup
        scene = {"example": "unchanged", "pose": "old_pose", "speed": "old_speed"}

        # Call method
        scene.update(self.trajectory_state)

        self.assertEqual(
            scene,
            {
                "example": "unchanged",
                "pose": [self.pose_x, self.pose_y, self.pose_heading],
                "speed": self.speed,
                "velocity_2d": self.velocity_2d,
                "lateral": self.lateral,
                "acceleration": self.acceleration,
            },
        )


if __name__ == "__main__":
    unittest.main()
