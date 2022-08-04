import unittest
from unittest.mock import Mock, patch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.utils.color import Color
from nuplan.planning.utils.serialization.to_scene import (
    to_scene_trajectory_from_list_ego_state,
    to_scene_trajectory_from_list_waypoint,
    to_scene_trajectory_state_from_ego_state,
    to_scene_trajectory_state_from_waypoint,
)


class TestToScene(unittest.TestCase):
    """
    Test scene conversions in to_scene.py
    """

    def test_to_scene_trajectory_state_from_ego_state(self) -> None:
        """
        Tests conversion from ego state to trajectory state (scene class)
        """
        # Set up
        ego_state = Mock(spec=EgoState)

        ego_state.rear_axle = [1.12, 2.11, 0.29]

        ego_state.dynamic_car_state.speed = 1.23

        ego_state.dynamic_car_state.rear_axle_velocity_2d.x = 0.12
        ego_state.dynamic_car_state.rear_axle_velocity_2d.y = 0.54

        ego_state.dynamic_car_state.rear_axle_acceleration_2d.x = 0.32
        ego_state.dynamic_car_state.rear_axle_acceleration_2d.y = 0.43
        ego_state.tire_steering_angle = 0.21

        # Call method under test
        result = to_scene_trajectory_state_from_ego_state(ego_state)

        # Assertions
        self.assertEqual(result.pose, [1.12, 2.11, 0.29])
        self.assertEqual(result.speed, 1.23)
        self.assertEqual(result.velocity_2d, [0.12, 0.54])
        self.assertEqual(result.lateral, [0.0, 0.0])
        self.assertEqual(result.acceleration, [0.32, 0.43])
        self.assertEqual(result.tire_steering_angle, 0.21)

    @patch("nuplan.planning.utils.serialization.to_scene.to_scene_trajectory_state_from_ego_state")
    def test_to_scene_trajectory_from_list_ego_state(self, mock_to_trajectory_state: Mock) -> None:
        """
        Tests conversion of list of ego states to trajectory structure (scene class)
        """
        # Set up
        mock_to_trajectory_state.side_effect = lambda state: "t_" + state
        ego_states = ["s1", "s2"]
        color = Color(0.5, 0.2, 0.5, 1)

        # Call method under test
        result = to_scene_trajectory_from_list_ego_state(ego_states, color)

        # Assertions
        self.assertEqual(result.color.to_list(), [0.5, 0.2, 0.5, 1])
        self.assertEqual(result.states, ["t_s1", "t_s2"])

    def test_to_scene_trajectory_state_from_waypoint(self) -> None:
        """
        Tests conversion from waypoint to trajectory state (scene class)
        """
        # Set up
        waypoint = Mock(spec=Waypoint)
        waypoint.center = [1.12, 2.11, 0.29]
        waypoint.velocity.magnitude.return_value = 1.23
        waypoint.velocity.x = 0.12
        waypoint.velocity.y = 0.54

        # Call method under test
        result = to_scene_trajectory_state_from_waypoint(waypoint)

        # Assertions
        self.assertEqual(result.pose, [1.12, 2.11, 0.29])
        self.assertEqual(result.speed, 1.23)
        self.assertEqual(result.velocity_2d, [0.12, 0.54])
        self.assertEqual(result.lateral, [0.0, 0.0])
        self.assertEqual(result.acceleration, None)
        self.assertEqual(result.tire_steering_angle, None)

    @patch("nuplan.planning.utils.serialization.to_scene.to_scene_trajectory_state_from_waypoint")
    def test_to_scene_trajectory_from_list_waypoint(self, mock_to_trajectory_state: Mock) -> None:
        """
        Tests conversion of list of waypoints to trajectory structure (scene class)
        """
        # Set up
        mock_to_trajectory_state.side_effect = lambda state: "w_" + state
        waypoints = ["s1", "s2"]
        color = Color(0.5, 0.2, 0.5, 1)

        # Call method under test
        result = to_scene_trajectory_from_list_waypoint(waypoints, color)

        # Assertions
        self.assertEqual(result.color.to_list(), [0.5, 0.2, 0.5, 1])
        self.assertEqual(result.states, ["w_s1", "w_s2"])


if __name__ == "__main__":
    unittest.main()
