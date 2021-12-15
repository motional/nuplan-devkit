import unittest

import numpy as np
from nuplan.common.actor_state.agent import Agent, AgentType, PredictedTrajectory, Waypoint
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.test.test_utils import get_sample_agent, get_sample_oriented_box


class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_token = 'abc123'
        self.agent_type = AgentType.VEHICLE
        self.sample_pose = StateSE2(1.0, 2.0, np.pi / 2.0)
        self.wlh = (2.0, 4.0, 1.5)
        self.velocity = StateVector2D(1.0, 2.2)

    def test_agent_types(self) -> None:
        """ Test that enum works for both existing and missing keys"""
        self.assertEqual(AgentType('vehicle'), AgentType.VEHICLE)
        self.assertEqual(AgentType('missing_key'), AgentType.UNKNOWN)

    def test_construction(self) -> None:
        """ Test that agents can be constructed correctly. """
        oriented_box = get_sample_oriented_box()
        agent = Agent(token=self.sample_token, agent_type=self.agent_type, oriented_box=oriented_box,
                      velocity=self.velocity)
        self.assertTrue(agent.angular_velocity is None)

    def test_set_predictions(self) -> None:
        """ Tests assignment of predictions to agents, and that this fails if the probabilities don't sum to one. """
        agent = get_sample_agent()
        waypoints = [Waypoint(TimePoint(t), get_sample_oriented_box(), StateVector2D(0.0, 0.0)) for t in range(5)]
        predictions = [PredictedTrajectory(0.3, waypoints), PredictedTrajectory(0.7, waypoints)]
        agent.predictions = predictions

        self.assertEqual(len(agent.predictions), 2)
        self.assertEqual(0.3, agent.predictions[0].probability)
        self.assertEqual(0.7, agent.predictions[1].probability)

        # Check that we fail to assign the predictions if the sum of probabilities is not one
        predictions += predictions
        with self.assertRaises(ValueError):
            agent.predictions = predictions


if __name__ == '__main__':
    unittest.main()
