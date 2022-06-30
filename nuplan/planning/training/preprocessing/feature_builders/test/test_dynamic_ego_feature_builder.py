import unittest

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractMap, MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.dynamic_ego_feature_builder import DynamicEgoFeatureBuilder
from nuplan.planning.training.preprocessing.features.dynamic_ego_feature import DynamicEgoFeature


class TestDynamicEgoFeatureBuilder(unittest.TestCase):
    """
    Tests DynamicEgoFeatureBuilder
    """

    def setUp(self) -> None:
        """
        Initializes feature builder
        """
        self.batch_size = 1
        self.past_time_horizon = 4.0
        self.num_agents = 10
        self.num_past_poses = 4
        self.num_total_past_poses = self.num_past_poses + 1  # past + present

        self.feature_builder = DynamicEgoFeatureBuilder(
            TrajectorySampling(num_poses=self.num_past_poses, time_horizon=self.past_time_horizon)
        )

    def test_dynamic_ego_feature_builder(self) -> None:
        """
        Test DynamicEgoFeatureBuilder
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents)
        feature = self.feature_builder.get_features_from_scenario(scenario)

        self.assertEqual(type(feature), DynamicEgoFeature)

        self.assertEqual(feature.batch_size, self.batch_size)

        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), DynamicEgoFeature.ego_state_dim())

    def test_get_feature_from_simulation(self) -> None:
        """
        Test get feature from simulation
        """
        # Build test scenario
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents)

        mock_meta_data = PlannerInitialization(
            map_api=MockAbstractMap(),
            expert_goal_state=StateSE2(0, 0, 0),
            route_roadblock_ids=None,
            mission_goal=StateSE2(0, 0, 0),
        )

        ego_past_states = scenario.get_ego_past_trajectory(iteration=0, num_samples=5, time_horizon=5)
        ego_initial_state = scenario.initial_ego_state
        ego_history = ego_past_states + [ego_initial_state]

        past_observations = scenario.get_past_tracked_objects(iteration=0, num_samples=5, time_horizon=5)
        initial_observation = scenario.initial_tracked_objects
        observation_history = past_observations + [initial_observation]

        history = SimulationHistoryBuffer.initialize_from_list(len(ego_history), ego_history, observation_history)
        current_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=scenario.get_traffic_light_status_at_iteration(0),
        )

        # Test scenario feature extraction
        feature = self.feature_builder.get_features_from_simulation(
            current_input=current_input, initialization=mock_meta_data
        )

        self.assertEqual(type(feature), DynamicEgoFeature)

        self.assertEqual(feature.batch_size, self.batch_size)

        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), DynamicEgoFeature.ego_state_dim())


if __name__ == '__main__':
    unittest.main()
