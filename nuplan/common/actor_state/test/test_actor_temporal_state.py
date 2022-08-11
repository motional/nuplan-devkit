import unittest
from typing import List, Optional
from unittest.mock import Mock

from nuplan.common.actor_state.agent import PredictedTrajectory
from nuplan.common.actor_state.agent_temporal_state import AgentTemporalState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.waypoint import Waypoint


class TestActorTemporalState(unittest.TestCase):
    """Test suite for the AgentTemporalState class"""

    def setUp(self) -> None:
        """Setup initial waypoints."""
        self.current_time_us = int(10 * 1e6)
        mock_oriented_box = Mock()
        self.future_waypoints: List[Optional[Waypoint]] = [
            Waypoint(time_point=TimePoint(self.current_time_us), oriented_box=mock_oriented_box),
            Waypoint(time_point=TimePoint(self.current_time_us + int(1e6)), oriented_box=mock_oriented_box),
        ]
        self.past_waypoints: List[Optional[Waypoint]] = [
            Waypoint(time_point=TimePoint(self.current_time_us - int(1e6)), oriented_box=mock_oriented_box),
            Waypoint(time_point=TimePoint(self.current_time_us), oriented_box=mock_oriented_box),
        ]

    def test_past_setting_successful(self) -> None:
        """Test that we can set past trajectory."""
        past_waypoints = [None] + self.past_waypoints
        actor = AgentTemporalState(
            initial_time_stamp=TimePoint(self.current_time_us),
            past_trajectory=PredictedTrajectory(waypoints=past_waypoints, probability=1.0),
        )
        self.assertEqual(actor.past_trajectory.probability, 1.0)
        self.assertEqual(len(actor.past_trajectory.valid_waypoints), 2)
        self.assertEqual(len(actor.past_trajectory), 3)
        # In this case since past_waypoints has size 2, first element is actually second last.
        self.assertEqual(actor.previous_state, self.past_waypoints[0])

    def test_past_setting_fail(self) -> None:
        """Test that we can raise if past trajectory does not start at current state."""
        past_waypoints = list(reversed(self.past_waypoints))
        with self.assertRaises(ValueError):
            # We raise because last state does not have the right time stamp
            AgentTemporalState(
                initial_time_stamp=TimePoint(self.current_time_us),
                past_trajectory=PredictedTrajectory(waypoints=past_waypoints, probability=1.0),
            )

    def test_future_trajectory_successful(self) -> None:
        """Test that we can set future predictions."""
        future_waypoints = self.future_waypoints

        actor = AgentTemporalState(
            initial_time_stamp=TimePoint(self.current_time_us),
            predictions=[PredictedTrajectory(waypoints=future_waypoints, probability=1.0)],
        )
        # Check that all variables are correctly set
        self.assertEqual(len(actor.predictions), 1)
        self.assertEqual(actor.predictions[0].probability, 1.0)

    def test_trajectory_successful_none(self) -> None:
        """Test that we can set future predictions with None."""
        actor = AgentTemporalState(
            initial_time_stamp=TimePoint(self.current_time_us), predictions=None, past_trajectory=None
        )
        # Check that all variables are correctly set
        self.assertEqual(len(actor.predictions), 0)
        self.assertEqual(actor.past_trajectory, None)

    def test_future_trajectory_fail(self) -> None:
        """Test that we can set future predictions, but it will fail if all conditions are not met."""
        future_waypoints = self.future_waypoints
        with self.assertRaises(ValueError):
            AgentTemporalState(
                initial_time_stamp=TimePoint(self.current_time_us),
                predictions=[PredictedTrajectory(waypoints=future_waypoints, probability=0.4)],
            )


if __name__ == '__main__':
    unittest.main()
