import unittest

import numpy as np
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder, _create_scenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import FeatureBuilderMetaData, \
    VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap


class TestVectorMapFeatureBuilder(unittest.TestCase):

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario_builder = NuPlanScenarioBuilder(version="nuplan_v0.1_mini", data_root="/data/sets/nuplan")
        self.scenario = _create_scenario(self.scenario_builder._db,
                                         ("unknown", self.scenario_builder._db.lidar_pc[10000].token),
                                         ScenarioMapping({}),
                                         get_pacifica_parameters())

    def test_vector_map_feature_builder(self) -> None:
        """
        Test VectorMapFeatureBuilder
        """
        feature_builder = VectorMapFeatureBuilder(radius=20, connection_scales=[2])
        self.assertEqual(feature_builder.get_feature_type(), VectorMap)

        features = feature_builder.get_features_from_scenario(self.scenario)
        self.assertEqual(type(features), VectorMap)

        ego_state = self.scenario.initial_ego_state
        detections = self.scenario.initial_detections
        meta_data = FeatureBuilderMetaData(self.scenario.map_api, self.scenario.get_mission_goal(), ego_state)
        features_sim = feature_builder.get_features_from_simulation([ego_state], [detections], meta_data)

        self.assertEqual(type(features_sim), VectorMap)
        self.assertTrue(np.allclose(features_sim.coords[0], features.coords[0], atol=1e-4))

        for connections, connections_simulation in zip(features_sim.multi_scale_connections[0].values(),
                                                       features.multi_scale_connections[0].values()):
            self.assertTrue(np.allclose(connections, connections_simulation))


if __name__ == '__main__':
    unittest.main()
