import unittest

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractMap, MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents


class TestAgentsFeatureBuilder(unittest.TestCase):
    """Test builder that constructs agent features during training and simulation."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.batch_size = 1
        self.past_time_horizon = 4.0
        self.num_agents = 10
        self.num_past_poses = 4
        self.num_total_past_poses = self.num_past_poses + 1  # past + present

        self.feature_builder = AgentsFeatureBuilder(
            TrajectorySampling(num_poses=self.num_past_poses, time_horizon=self.past_time_horizon)
        )

    def test_agent_feature_builder(self) -> None:
        """
        Test AgentFeatureBuilder
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents)
        feature = self.feature_builder.get_features_from_scenario(scenario)

        self.assertEqual(type(feature), Agents)

        self.assertEqual(feature.batch_size, self.batch_size)

        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), Agents.ego_state_dim())

        self.assertEqual(len(feature.agents), self.batch_size)
        self.assertEqual(len(feature.agents[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.agents[0][0]), self.num_agents)
        self.assertEqual(len(feature.agents[0][0][0]), Agents.agents_states_dim())

    def test_no_agents(self) -> None:
        """
        Test when there are no agents
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=0)
        feature = self.feature_builder.get_features_from_scenario(scenario)

        self.assertEqual(type(feature), Agents)

        self.assertEqual(feature.batch_size, self.batch_size)

        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), Agents.ego_state_dim())

        self.assertEqual(len(feature.agents), self.batch_size)
        self.assertEqual(len(feature.agents[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.agents[0][0]), 0)  # no agents
        self.assertEqual(feature.agents[0].shape[1], 0)
        self.assertEqual(feature.agents[0].shape[2], Agents.agents_states_dim())

    def test_get_feature_from_simulation(self) -> None:
        """
        Test get feature from simulation
        """
        scenario = MockAbstractScenario(number_of_past_iterations=10, number_of_detections=self.num_agents)

        mock_meta_data = PlannerInitialization(
            map_api=MockAbstractMap(),
            expert_goal_state=StateSE2(0, 0, 0),
            route_roadblock_ids=None,
            mission_goal=StateSE2(0, 0, 0),
        )

        ego_past_states = scenario.get_ego_past_trajectory(iteration=0, num_samples=10, time_horizon=5)
        ego_initial_state = scenario.initial_ego_state
        ego_history = ego_past_states + [ego_initial_state]

        past_observations = scenario.get_past_tracked_objects(iteration=0, num_samples=10, time_horizon=5)
        initial_observation = scenario.initial_tracked_objects
        observation_history = past_observations + [initial_observation]

        history = SimulationHistoryBuffer.initialize_from_list(
            len(ego_history), ego_history, observation_history, scenario.database_interval
        )
        current_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=scenario.get_traffic_light_status_at_iteration(0),
        )

        feature = self.feature_builder.get_features_from_simulation(
            current_input=current_input, initialization=mock_meta_data
        )

        self.assertEqual(type(feature), Agents)

        self.assertEqual(feature.batch_size, self.batch_size)

        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), Agents.ego_state_dim())

        self.assertEqual(len(feature.agents), self.batch_size)
        self.assertEqual(len(feature.agents[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.agents[0][0]), self.num_agents)
        self.assertEqual(len(feature.agents[0][0][0]), Agents.agents_states_dim())


if __name__ == '__main__':
    unittest.main()
