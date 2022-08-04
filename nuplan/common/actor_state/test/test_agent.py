import unittest

import numpy as np

from nuplan.common.actor_state.agent import Agent, PredictedTrajectory
from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.test.test_utils import get_sample_agent, get_sample_oriented_box
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.waypoint import Waypoint


class TestAgent(unittest.TestCase):
    """Test suite for the Agent class"""

    def setUp(self) -> None:
        """Setup parameters for tests"""
        self.sample_token = 'abc123'
        self.track_token = 'abc123'
        self.timestamp = 123
        self.agent_type = TrackedObjectType.VEHICLE
        self.sample_pose = StateSE2(1.0, 2.0, np.pi / 2.0)
        self.wlh = (2.0, 4.0, 1.5)
        self.velocity = StateVector2D(1.0, 2.2)

    def test_agent_state(self) -> None:
        """Test AgentState."""
        angular_velocity = 10.0
        oriented_box = get_sample_oriented_box()
        metadata = SceneObjectMetadata(
            token=self.sample_token, track_token=self.track_token, timestamp_us=self.timestamp, track_id=None
        )
        agent_state = AgentState(
            TrackedObjectType.VEHICLE,
            oriented_box=oriented_box,
            velocity=self.velocity,
            metadata=metadata,
            angular_velocity=angular_velocity,
        )
        self.assertEqual(agent_state.tracked_object_type, TrackedObjectType.VEHICLE)
        self.assertEqual(agent_state.box, oriented_box)
        self.assertEqual(agent_state.metadata, metadata)
        self.assertEqual(agent_state.velocity, self.velocity)
        self.assertEqual(agent_state.angular_velocity, angular_velocity)
        self.assertEqual(agent_state.token, metadata.token)
        self.assertEqual(agent_state.track_token, metadata.track_token)

    def test_agent_types(self) -> None:
        """Test that enum works for both existing and missing keys"""
        self.assertEqual(TrackedObjectType(0), TrackedObjectType.VEHICLE)
        self.assertEqual(TrackedObjectType.VEHICLE.fullname, "vehicle")
        with self.assertRaises(ValueError):
            TrackedObjectType('missing_key')

    def test_construction(self) -> None:
        """Test that agents can be constructed correctly."""
        oriented_box = get_sample_oriented_box()
        agent = Agent(
            metadata=SceneObjectMetadata(
                token=self.sample_token, track_token=self.track_token, timestamp_us=self.timestamp, track_id=None
            ),
            tracked_object_type=self.agent_type,
            oriented_box=oriented_box,
            velocity=self.velocity,
        )
        self.assertTrue(agent.angular_velocity is None)

    def test_set_predictions(self) -> None:
        """Tests assignment of predictions to agents, and that this fails if the probabilities don't sum to one."""
        agent = get_sample_agent()
        waypoints = [Waypoint(TimePoint(t), get_sample_oriented_box(), StateVector2D(0.0, 0.0)) for t in range(5)]
        predictions = [
            PredictedTrajectory(0.3, waypoints),
            PredictedTrajectory(0.7, waypoints),
        ]
        agent.predictions = predictions

        self.assertEqual(len(agent.predictions), 2)
        self.assertEqual(0.3, agent.predictions[0].probability)
        self.assertEqual(0.7, agent.predictions[1].probability)

        # Check that we fail to assign the predictions if the sum of probabilities is not one
        predictions += predictions
        with self.assertRaises(ValueError):
            agent.predictions = predictions

    def test_set_past_trajectory(self) -> None:
        """Tests assignment of past trajectory to agents."""
        agent = get_sample_agent()
        waypoints = [
            Waypoint(TimePoint(t), get_sample_oriented_box(), StateVector2D(0.0, 0.0))
            for t in range(agent.metadata.timestamp_us + 1)
        ]
        agent.past_trajectory = PredictedTrajectory(1, waypoints)

        self.assertEqual(len(agent.past_trajectory.waypoints), 11)

        with self.assertRaises(ValueError):
            # Fail because the final state does not land at current ego's position
            agent.past_trajectory = PredictedTrajectory(
                1, [Waypoint(TimePoint(t), get_sample_oriented_box(), StateVector2D(0.0, 0.0)) for t in range(3)]
            )


if __name__ == '__main__':
    unittest.main()
