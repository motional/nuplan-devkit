import copy
import unittest
from typing import Dict, List

import torch

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractMap, MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import (
    GenericAgentsFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex, EgoInternalIndex


class TestGenericAgentsFeatureBuilder(unittest.TestCase):
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
        self.agent_features = [
            'VEHICLE',
            'PEDESTRIAN',
            'BICYCLE',
            'TRAFFIC_CONE',
            'BARRIER',
            'CZONE_SIGN',
            'GENERIC_OBJECT',
        ]

        self.tracked_object_types: List[TrackedObjectType] = []
        for feature_name in self.agent_features:
            try:
                self.tracked_object_types.append(TrackedObjectType[feature_name])
            except KeyError:
                raise ValueError(f"Object representation for layer: {feature_name} is unavailable!")

        self.feature_builder = GenericAgentsFeatureBuilder(
            self.agent_features, TrajectorySampling(num_poses=self.num_past_poses, time_horizon=self.past_time_horizon)
        )

    def test_generic_agent_feature_builder(self) -> None:
        """
        Test GenericAgentFeatureBuilder
        """
        scenario = MockAbstractScenario(
            number_of_past_iterations=10,
            number_of_detections=self.num_agents,
            tracked_object_types=self.tracked_object_types,
        )
        feature = self.feature_builder.get_features_from_scenario(scenario)

        self.assertEqual(type(feature), GenericAgents)

        self.assertEqual(feature.batch_size, self.batch_size)

        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())

        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_no_agents(self) -> None:
        """
        Test when there are no agents
        """
        scenario = MockAbstractScenario(
            number_of_past_iterations=10, number_of_detections=0, tracked_object_types=self.tracked_object_types
        )
        feature = self.feature_builder.get_features_from_scenario(scenario)

        self.assertEqual(type(feature), GenericAgents)

        self.assertEqual(feature.batch_size, self.batch_size)

        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())

        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), 0)  # no agents
            self.assertEqual(feature.agents[feature_name][0].shape[1], 0)
            self.assertEqual(feature.agents[feature_name][0].shape[2], GenericAgents.agents_states_dim())

    def test_get_feature_from_simulation(self) -> None:
        """
        Test get feature from simulation
        """
        scenario = MockAbstractScenario(
            number_of_past_iterations=10,
            number_of_detections=self.num_agents,
            tracked_object_types=self.tracked_object_types,
        )

        mock_meta_data = PlannerInitialization(
            map_api=MockAbstractMap(),
            route_roadblock_ids=None,
            mission_goal=StateSE2(0, 0, 0),
        )

        ego_past_states = list(scenario.get_ego_past_trajectory(iteration=0, num_samples=10, time_horizon=5))
        ego_initial_state = scenario.initial_ego_state
        ego_history = ego_past_states + [ego_initial_state]

        past_observations = list(scenario.get_past_tracked_objects(iteration=0, num_samples=10, time_horizon=5))
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

        self.assertEqual(type(feature), GenericAgents)

        self.assertEqual(feature.batch_size, self.batch_size)

        self.assertEqual(len(feature.ego), self.batch_size)
        self.assertEqual(len(feature.ego[0]), self.num_total_past_poses)
        self.assertEqual(len(feature.ego[0][0]), GenericAgents.ego_state_dim())

        for feature_name in self.agent_features:
            self.assertTrue(feature_name in feature.agents)
            self.assertEqual(len(feature.agents[feature_name]), self.batch_size)
            self.assertEqual(len(feature.agents[feature_name][0]), self.num_total_past_poses)
            self.assertEqual(len(feature.agents[feature_name][0][0]), self.num_agents)
            self.assertEqual(len(feature.agents[feature_name][0][0][0]), GenericAgents.agents_states_dim())

    def test_agents_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the Generic Agents Feature Builder scripts properly
        """
        config = self.feature_builder.precomputed_feature_config()
        for expected_key in ["past_ego_states", "past_time_stamps"]:
            self.assertTrue(expected_key in config)

            config_dict = config[expected_key]
            self.assertTrue(len(config_dict) == 3)
            self.assertEqual(0, int(config_dict["iteration"]))
            self.assertEqual(self.num_past_poses, int(config_dict["num_samples"]))
            self.assertEqual(self.past_time_horizon, int(float(config_dict["time_horizon"])))

        tracked_objects_config_dict = config["past_tracked_objects"]
        self.assertTrue(len(tracked_objects_config_dict) == 4)
        self.assertEqual(0, int(tracked_objects_config_dict["iteration"]))
        self.assertEqual(self.num_past_poses, int(tracked_objects_config_dict["num_samples"]))
        self.assertEqual(self.past_time_horizon, int(float(tracked_objects_config_dict["time_horizon"])))
        self.assertTrue("agent_features" in tracked_objects_config_dict)
        self.assertEqual(",".join(self.agent_features), tracked_objects_config_dict["agent_features"])

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
        list_tensor_data = {
            f"past_tracked_objects.{feature_name}": past_tracked_objects for feature_name in self.agent_features
        }
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
                torch.testing.assert_allclose(py, scripted, atol=0.05, rtol=0.05)

        self.assertEqual(0, len(scripted_list_list_tensors))
        self.assertEqual(0, len(py_list_list_tensors))


if __name__ == '__main__':
    unittest.main()
