import unittest
from typing import Any, Dict, List

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.utils.serialization.scene_simple_trajectory import SceneSimpleTrajectory


class TestSceneSimpleTrajectory(unittest.TestCase):
    """
    Tests the class SceneSimpleTrajectory
    """

    def setUp(self) -> None:
        """
        Sets up for the test cases
        """
        state1: Dict[str, Any] = {'timestamp': 1, 'pose': [1, 2, 3]}
        state2: Dict[str, Any] = {'timestamp': 2, 'pose': [3, 4, 5]}
        prediction_states: List[Dict[str, Any]] = [state1, state2]
        self.width = 3
        self.length = 6
        self.height = 2
        self.scene_simple_trajectory = SceneSimpleTrajectory(
            prediction_states, width=self.width, length=self.length, height=self.height
        )

    def test_init(self) -> None:
        """
        Tests the init of SceneSiimpleTrajectory
        """
        # Setup
        state1: Dict[str, Any] = {'timestamp': 1, 'pose': [1, 2, 3]}
        state2: Dict[str, Any] = {'timestamp': 2, 'pose': [3, 4, 5]}
        prediction_states: List[Dict[str, Any]] = [state1, state2]

        # Call the method under test
        result = SceneSimpleTrajectory(prediction_states, width=self.width, length=self.length, height=self.height)

        # Assertions
        self.assertEqual(result._start_time, 1)
        self.assertEqual(result._end_time, 2)

    def test_start_time(self) -> None:
        """
        Tests the start time property
        """
        # Setup
        scene_simple_trajectory = self.scene_simple_trajectory

        # Call the method under test
        result = scene_simple_trajectory.start_time

        # Assertions
        self.assertEqual(result, 1)

    def test_end_time(self) -> None:
        """
        Tests the start time property
        """
        # Setup
        scene_simple_trajectory = self.scene_simple_trajectory

        # Call the method under test
        result = scene_simple_trajectory.end_time

        # Assertions
        self.assertEqual(result, 2)

    def test_get_state_at_time(self) -> None:
        """
        Tests the get state at time method
        """
        # Setup
        scene_simple_trajectory = self.scene_simple_trajectory

        # Call the method under test
        result = scene_simple_trajectory.get_state_at_time(TimePoint(int(1e6)))

        # Assertions
        self.assertEqual(result.x, 1)
        self.assertEqual(result.y, 2)

    def test_get_sampled_trajectory(self) -> None:
        """
        Tests the get sampled method
        """
        # Setup
        scene_simple_trajectory = self.scene_simple_trajectory

        # Call the method under test
        result = scene_simple_trajectory.get_sampled_trajectory()

        # Assertions
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].x, 1)
        self.assertEqual(result[1].x, 3)


if __name__ == "__main__":
    unittest.main()
