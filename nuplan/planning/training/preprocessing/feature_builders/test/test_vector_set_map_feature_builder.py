import unittest
from typing import Dict, List

import numpy as np
import torch

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap


class TestVectorSetMapFeatureBuilder(unittest.TestCase):
    """Test feature builder that constructs map features in vector set format."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario = MockAbstractScenario()
        self.batch_size = 1
        self.radius = 35
        self.interpolation_method = 'linear'
        self.map_features = [
            'LANE',
            'LEFT_BOUNDARY',
            'RIGHT_BOUNDARY',
            'STOP_LINE',
            'CROSSWALK',
            'ROUTE_LANES',
        ]
        self.max_elements = {
            'LANE': 30,
            'LEFT_BOUNDARY': 30,
            'RIGHT_BOUNDARY': 30,
            'STOP_LINE': 20,
            'CROSSWALK': 20,
            'ROUTE_LANES': 30,
        }
        self.max_points = {
            'LANE': 20,
            'LEFT_BOUNDARY': 20,
            'RIGHT_BOUNDARY': 20,
            'STOP_LINE': 20,
            'CROSSWALK': 20,
            'ROUTE_LANES': 20,
        }

        self.feature_builder = VectorSetMapFeatureBuilder(
            map_features=self.map_features,
            max_elements=self.max_elements,
            max_points=self.max_points,
            radius=self.radius,
            interpolation_method=self.interpolation_method,
        )

    def test_vector_set_map_feature_builder(self) -> None:
        """
        Tests VectorSetMapFeatureBuilder.
        """
        self.assertEqual(self.feature_builder.get_feature_type(), VectorSetMap)

        features = self.feature_builder.get_features_from_scenario(self.scenario)
        self.assertEqual(type(features), VectorSetMap)
        self.assertEqual(features.batch_size, self.batch_size)

        for feature_name in self.map_features:
            self.assertEqual(
                features.coords[feature_name][0].shape,
                (self.max_elements[feature_name], self.max_points[feature_name], 2),
            )
            self.assertEqual(
                features.availabilities[feature_name][0].shape,
                (self.max_elements[feature_name], self.max_points[feature_name]),
            )

            # check data is specified as padded
            np.testing.assert_array_equal(
                features.availabilities[feature_name][0],
                np.zeros((self.max_elements[feature_name], self.max_points[feature_name]), dtype=np.bool_),
            )

    def test_vector_set_map_feature_builder_simulation_and_scenario_match(self) -> None:
        """
        Tests that get_features_from_scenario and get_features_from_simulation give same results.
        """
        features = self.feature_builder.get_features_from_scenario(self.scenario)

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

        features_sim = self.feature_builder.get_features_from_simulation(
            current_input=current_input, initialization=meta_data
        )

        self.assertEqual(type(features_sim), VectorSetMap)

        self.assertEqual(set(features_sim.coords.keys()), set(features.coords.keys()))
        for feature_name in features_sim.coords.keys():
            np.testing.assert_allclose(
                features_sim.coords[feature_name][0], features.coords[feature_name][0], atol=1e-4
            )

        self.assertEqual(set(features_sim.traffic_light_data.keys()), set(features.traffic_light_data.keys()))
        for feature_name in features_sim.traffic_light_data.keys():
            np.testing.assert_allclose(
                features_sim.traffic_light_data[feature_name][0], features.traffic_light_data[feature_name][0]
            )

        self.assertEqual(set(features_sim.availabilities.keys()), set(features.availabilities.keys()))
        for feature_name in features_sim.availabilities.keys():
            np.testing.assert_array_equal(
                features_sim.availabilities[feature_name][0], features.availabilities[feature_name][0]
            )

    def test_vector_set_map_feature_builder_scripts_properly(self) -> None:
        """
        Tests that the VectorSetMapFeatureBuilder can be scripted properly.
        """
        self.assertEqual(self.feature_builder.get_feature_type(), VectorSetMap)

        scripted_builder = torch.jit.script(self.feature_builder)
        self.assertIsNotNone(scripted_builder)

        # Assert that the feature config is exported properly
        config = scripted_builder.precomputed_feature_config()
        self.assertTrue("initial_ego_state" in config)
        self.assertTrue("neighbor_vector_set_map" in config)
        self.assertTrue("radius" in config["neighbor_vector_set_map"])
        self.assertEqual(str(self.radius), config["neighbor_vector_set_map"]["radius"])
        self.assertEqual(str(self.interpolation_method), config["neighbor_vector_set_map"]["interpolation_method"])
        self.assertEqual(",".join(self.map_features), config["neighbor_vector_set_map"]["map_features"])
        max_elements: List[str] = [
            f"{feature_name}.{feature_max_elements}" for feature_name, feature_max_elements in self.max_elements.items()
        ]
        max_points: List[str] = [
            f"{feature_name}.{feature_max_points}" for feature_name, feature_max_points in self.max_points.items()
        ]
        self.assertEqual(",".join(max_elements), config["neighbor_vector_set_map"]["max_elements"])
        self.assertEqual(",".join(max_points), config["neighbor_vector_set_map"]["max_points"])

        # Assert that the scriptable method works.
        tensor_data = {"anchor_state": torch.zeros((3,))}
        for feature_name in self.map_features:
            feature_max_elements = self.max_elements[feature_name]
            feature_max_points = self.max_points[feature_name]
            tensor_data[f"coords.{feature_name}"] = torch.rand(
                (feature_max_elements, feature_max_points, 2), dtype=torch.float64
            )
            tensor_data[f"traffic_light_data.{feature_name}"] = torch.zeros(
                (feature_max_elements, feature_max_points, 4), dtype=torch.int64
            )
            tensor_data[f"availabilities.{feature_name}"] = torch.zeros(
                (feature_max_elements, feature_max_points), dtype=torch.bool
            )

        list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]] = {}

        scripted_tensor_output, scripted_list_output, scripted_list_list_output = scripted_builder.scriptable_forward(
            tensor_data, list_tensor_data, list_list_tensor_data
        )
        py_tensor_output, py_list_output, py_list_list_output = self.feature_builder.scriptable_forward(
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
