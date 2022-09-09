import unittest

from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layers_utils import StateIndex
from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex


class TestKinematicBicycleLayersUtils(unittest.TestCase):
    """
    Test Kinematic Bicycle Layers utils.
    """

    def setUp(self) -> None:
        """Sets variables for testing"""
        pass

    def test_enums_equal(self) -> None:
        """
        Ensure our internal indexing matches the
        one from Agents' trajectories
        """
        self.assertEqual(StateIndex.X_POS, AgentFeatureIndex.x())
        self.assertEqual(StateIndex.Y_POS, AgentFeatureIndex.y())
        self.assertEqual(StateIndex.YAW, AgentFeatureIndex.heading())
        self.assertEqual(StateIndex.X_VELOCITY, AgentFeatureIndex.vx())
        self.assertEqual(StateIndex.Y_VELOCITY, AgentFeatureIndex.vy())
        self.assertEqual(StateIndex.YAW_RATE, AgentFeatureIndex.yaw_rate())


if __name__ == '__main__':
    unittest.main()
