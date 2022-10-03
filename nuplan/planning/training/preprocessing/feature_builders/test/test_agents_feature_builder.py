import copy
import unittest
from typing import Dict, List

import torch

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractMap, MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex, EgoInternalIndex


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

    def test_agents_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the Agents Feature Builder scripts properly
        """
        config = self.feature_builder.precomputed_feature_config()
        for expected_key in ["past_ego_states", "past_time_stamps", "past_tracked_objects"]:
            self.assertTrue(expected_key in config)

            config_dict = config[expected_key]
            self.assertTrue(len(config_dict) == 3)
            self.assertEqual(0, int(config_dict["iteration"]))
            self.assertEqual(self.num_past_poses, int(config_dict["num_samples"]))
            self.assertEqual(self.past_time_horizon, int(float(config_dict["time_horizon"])))

        # Create some mock data
        num_frames = 5
        num_agents = 3
        ego_dim = EgoInternalIndex.dim()
        agent_dim = AgentInternalIndex.dim()
        past_ego_states = torch.zeros((num_frames, ego_dim), dtype=torch.float32)
        past_timestamps = torch.tensor([i * 50 for i in range(num_frames)], dtype=torch.int64)
        past_tracked_objects = [torch.ones((num_agents, agent_dim), dtype=torch.float32) for _ in range(num_frames)]
        for i in range(num_frames):
            for j in range(num_agents):
                past_tracked_objects[i][j, :] *= j + 1

        tensor_data = {"past_ego_states": past_ego_states, "past_time_stamps": past_timestamps}
        list_tensor_data = {"past_tracked_objects": past_tracked_objects}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}

        scripted_builder = torch.jit.script(self.feature_builder)

        scripted_tensors, scripted_list_tensors, scripted_list_list_tensors = scripted_builder.scriptable_forward(
            copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data)
        )

        py_tensors, py_list_tensors, py_list_list_tensors = self.feature_builder.scriptable_forward(
            copy.deepcopy(tensor_data), copy.deepcopy(list_tensor_data), copy.deepcopy(list_list_tensor_data)
        )

        self.assertEqual(0, len(scripted_tensors))
        self.assertEqual(0, len(py_tensors))

        self.assertEqual(len(scripted_list_tensors), len(py_list_tensors))
        for key in py_list_tensors:
            scripted_list = scripted_list_tensors[key]
            py_list = py_list_tensors[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                scripted = scripted_list[i]
                py = py_list[i]
                torch.testing.assert_allclose(py, scripted, atol=0.01, rtol=0.01)

        self.assertEqual(0, len(scripted_list_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))


if __name__ == '__main__':
    unittest.main()
