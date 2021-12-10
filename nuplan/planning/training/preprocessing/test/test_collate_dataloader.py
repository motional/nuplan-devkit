import unittest

import torch.utils.data
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_caching_preprocessor import FeatureCachingPreprocessor
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import \
    EgoTrajectoryTargetBuilder
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

NUM_BATCHES = 20


class TestCollateDataLoader(unittest.TestCase):
    """
    Tests data loading functionality
    """

    def setUp(self) -> None:
        """
        Initializes
        """
        self.batch_size = 4

        self.scenario_builder = NuPlanScenarioBuilder(version="nuplan_v0.1_mini", data_root="/data/sets/nuplan")

        features = FeatureCachingPreprocessor(cache_dir=None,
                                              feature_builders=[
                                                  RasterFeatureBuilder(
                                                      map_features={
                                                          'LANE': 255, 'INTERSECTION': 255,
                                                          'STOP_LINE': 128, 'CROSSWALK': 128},
                                                      num_input_channels=4,
                                                      target_width=224,
                                                      target_height=224,
                                                      target_pixel_size=0.5,
                                                      ego_width=2.297,
                                                      ego_front_length=4.049,
                                                      ego_rear_length=1.127,
                                                      ego_longitudinal_offset=0.0,
                                                      baseline_path_thickness=1,
                                                  ),
                                                  VectorMapFeatureBuilder(radius=20)
                                              ],
                                              target_builders=[
                                                  EgoTrajectoryTargetBuilder(
                                                      TrajectorySampling(time_horizon=6.0, num_poses=12))
                                              ],
                                              force_feature_computation=False)

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilters(
            log_names=None,
            log_labels=None,
            max_scenarios_per_log=None,
            scenario_types=None,
            scenario_tokens=None,
            map_name=None,
            shuffle=False,
            limit_scenarios_per_type=None,
            subsample_ratio=0.05,
            flatten_scenarios=True,
            remove_invalid_goals=True,
            limit_total_scenarios=20,
        ), Sequential())

        self.assertGreater(len(scenarios), 0)

        # Keep only a few scenarios instead of testing the whole extraction
        scenarios = [scenarios[0], scenarios[10], scenarios[-1]]

        dataset = ScenarioDataset(scenarios=scenarios, feature_caching_preprocessor=features)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=2,
                                                      pin_memory=False, drop_last=True, collate_fn=FeatureCollate())

    def test_dataloader(self) -> None:
        """
        Tests that the training dataloader can be iterated without errors
        """
        dataloader = self.dataloader
        dataloader_iter = iter(dataloader)
        iterations = min(len(dataloader), NUM_BATCHES)

        for _ in range(iterations):
            features, targets = next(dataloader_iter)
            self.assertTrue("vector_map" in features.keys())
            vector_map: VectorMap = features["vector_map"]
            self.assertEqual(vector_map.num_of_batches, self.batch_size)
            self.assertEqual(len(vector_map.coords), self.batch_size)
            self.assertEqual(len(vector_map.multi_scale_connections), self.batch_size)


if __name__ == '__main__':
    unittest.main()
