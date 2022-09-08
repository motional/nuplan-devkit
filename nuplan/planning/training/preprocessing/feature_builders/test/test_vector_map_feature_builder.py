import unittest
from typing import Dict, List

import numpy as np
import torch

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap


class TestVectorMapFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs map features in vectorized format."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        # TODO: Check for red light data when db is available
        self.scenario = get_test_nuplan_scenario()

    def test_vector_map_feature_builder(self) -> None:
        """
        Test VectorMapFeatureBuilder
        """
        feature_builder = VectorMapFeatureBuilder(radius=20, connection_scales=[2])
        self.assertEqual(feature_builder.get_feature_type(), VectorMap)

        features = feature_builder.get_features_from_scenario(self.scenario)
        self.assertEqual(type(features), VectorMap)

        ego_state = self.scenario.initial_ego_state
        detections = self.scenario.initial_tracked_objects
        meta_data = PlannerInitialization(
            map_api=self.scenario.map_api,
            mission_goal=self.scenario.get_mission_goal(),
            route_roadblock_ids=self.scenario.get_route_roadblock_ids(),
        )

        history = SimulationHistoryBuffer.initialize_from_list(
            1, [ego_state], [detections], self.scenario.database_interval
        )
        iteration = SimulationIteration(TimePoint(0), 0)
        tl_data = self.scenario.get_traffic_light_status_at_iteration(iteration.index)
        current_input = PlannerInput(iteration=iteration, history=history, traffic_light_data=tl_data)

        features_sim = feature_builder.get_features_from_simulation(
            current_input=current_input, initialization=meta_data
        )

        self.assertEqual(type(features_sim), VectorMap)
        self.assertTrue(np.allclose(features_sim.coords[0], features.coords[0], atol=1e-4))

        for connections, connections_simulation in zip(
            features_sim.multi_scale_connections[0].values(), features.multi_scale_connections[0].values()
        ):
            self.assertTrue(np.allclose(connections, connections_simulation))

        for lane in range(len(features_sim.lane_groupings[0])):
            for lane_groupings, lane_groupings_simulation in zip(
                features_sim.lane_groupings[0][lane], features.lane_groupings[0][lane]
            ):
                self.assertTrue(np.allclose(lane_groupings, lane_groupings_simulation))

        self.assertTrue(np.allclose(features_sim.on_route_status[0], features.on_route_status[0], atol=1e-4))

        self.assertTrue(np.allclose(features_sim.traffic_light_data[0], features.traffic_light_data[0]))

    def test_vector_map_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the VectorMapFeatureBuilder can be scripted properly.
        """
        feature_builder = VectorMapFeatureBuilder(radius=20, connection_scales=[2])
        self.assertEqual(feature_builder.get_feature_type(), VectorMap)

        scripted_builder = torch.jit.script(feature_builder)
        self.assertIsNotNone(scripted_builder)

        # Assert that the feature config is exported properly
        config = scripted_builder.precomputed_feature_config()
        self.assertTrue("initial_ego_state" in config)
        self.assertTrue("neighbor_vector_map" in config)
        self.assertTrue("radius" in config["neighbor_vector_map"])
        self.assertEqual("20", config["neighbor_vector_map"]["radius"])

        # Assert that the scriptable method works.
        num_lane_segment = 5
        num_connections = 7
        tensor_data = {
            "lane_segment_coords": torch.rand((num_lane_segment, 2, 2), dtype=torch.float64),
            "lane_segment_conns": torch.zeros((num_connections, 2), dtype=torch.int64),
            "on_route_status": torch.zeros((num_lane_segment, 2), dtype=torch.float32),
            "traffic_light_array": torch.zeros((num_lane_segment, 4), dtype=torch.float32),
            "anchor_state": torch.zeros((3,), dtype=torch.float64),
        }

        list_tensor_data = {
            "lane_segment_groupings": [torch.zeros(size=(2,), dtype=torch.int64) for _ in range(num_lane_segment)]
        }

        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}

        scripted_tensor_output, scripted_list_output, scripted_list_list_output = scripted_builder.scriptable_forward(
            tensor_data, list_tensor_data, list_list_tensor_data
        )
        py_tensor_output, py_list_output, py_list_list_output = feature_builder.scriptable_forward(
            tensor_data, list_tensor_data, list_list_tensor_data
        )

        self.assertEqual(0, len(scripted_tensor_output))
        self.assertEqual(0, len(py_tensor_output))

        self.assertEqual(len(scripted_list_output), len(py_list_output))
        for key in py_list_output:
            self.assertEqual(len(py_list_output[key]), len(scripted_list_output[key]))
            for i in range(len(py_list_output[key])):
                torch.testing.assert_close(py_list_output[key][i], scripted_list_output[key][i])

        self.assertEqual(len(py_list_list_output), len(scripted_list_list_output))
        for key in py_list_list_output:
            py_list = py_list_list_output[key]
            scripted_list = scripted_list_list_output[key]
            self.assertEqual(len(py_list), len(scripted_list))
            for i in range(len(py_list)):
                py = py_list[i]
                script = scripted_list[i]
                self.assertEqual(len(py), len(script))
                for j in range(len(py)):
                    torch.testing.assert_close(py[j], script[j])


if __name__ == '__main__':
    unittest.main()
