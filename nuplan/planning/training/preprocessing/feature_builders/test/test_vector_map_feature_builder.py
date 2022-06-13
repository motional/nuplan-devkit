import unittest

import numpy as np

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
            expert_goal_state=ego_state.rear_axle,
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


if __name__ == '__main__':
    unittest.main()
