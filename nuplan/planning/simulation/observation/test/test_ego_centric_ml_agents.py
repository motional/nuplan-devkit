import unittest
from unittest.mock import Mock, patch

import numpy as np

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.observation.ego_centric_ml_agents import EgoCentricMLAgents
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories


class TestEgoCentricMLAgents(unittest.TestCase):
    """Test important functions of EgoCentricMLAgents class"""

    def setUp(self) -> None:
        """Initialize scenario and model for constructing AgentCentricMLAgents class."""
        self.scenario = MockAbstractScenario(number_of_detections=1)
        self.model = Mock(spec=TorchModuleWrapper)
        self.model.future_trajectory_sampling = TrajectorySampling(num_poses=1, time_horizon=1.0)

        self.pred_trajectory = AgentsTrajectories(data=[np.array([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]])])

    def test_update_observation_with_predictions(self) -> None:
        """Test the _update_observation_with_predictions fucntion."""
        obs = EgoCentricMLAgents(model=self.model, scenario=self.scenario)
        obs.initialize()

        # Initialization
        self.assertEqual(len(obs._agents), 1)
        # This is the default agent from Mock Scenario
        self.assertEqual(obs._agents['0'].center.x, 1.0)
        self.assertEqual(obs._agents['0'].center.y, 2.0)
        self.assertAlmostEqual(obs._agents['0'].center.heading, np.pi / 2)

        # Update agents based on predictions
        # We did not call update_observation. Assume the time step is 1 second.
        obs.step_time = TimePoint(1e6)
        predictions = {'agents_trajectory': self.pred_trajectory}
        obs._update_observation_with_predictions(predictions)
        self.assertEqual(len(obs._agents), 1)
        # The predictions are relative to ego AV.
        self.assertAlmostEqual(obs._agents['0'].center.x, 1.0)
        self.assertAlmostEqual(obs._agents['0'].center.y, 1.0)
        self.assertAlmostEqual(obs._agents['0'].center.heading, 0.0)

    @patch('nuplan.planning.simulation.planner.ml_planner.model_loader.ModelLoader.infer')
    def test_infer_model(self, mock_infer: Mock) -> None:
        """Test _infer_model function."""
        predictions = {'agents_trajectory': self.pred_trajectory.to_feature_tensor()}
        mock_infer.return_value = predictions

        obs = EgoCentricMLAgents(model=self.model, scenario=self.scenario)
        obs.initialize()

        agents_raster = Mock(spec=Agents)
        features = {'agents': agents_raster}
        results = obs._infer_model(features)

        mock_infer.assert_called_with(features)
        self.assertIn(obs.prediction_type, results)
        self.assertIsInstance(results[obs.prediction_type], AgentsTrajectories)
        self.assertIsInstance(results[obs.prediction_type].data[0], np.ndarray)


if __name__ == "__main__":
    unittest.main()
