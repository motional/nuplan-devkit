import logging
import pathlib
import unittest
from typing import Any

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_caching_preprocessor import FeatureCachingPreprocessor
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import \
    EgoTrajectoryTargetBuilder
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


class TestFeatureCachingPreprocessor(unittest.TestCase):

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario_builder = NuPlanScenarioBuilder(version="nuplan_v0.1_mini", data_root="/data/sets/nuplan")
        self.cache_dir = pathlib.Path("/tmp/test")

    def test_sample(self) -> None:
        """
        Test computation of a features for sample
        """
        # Features
        raster_feature_builder = RasterFeatureBuilder(
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
        )
        vectormap_builder = VectorMapFeatureBuilder(radius=20)

        # Targets
        ego_trajectory_target_builder = EgoTrajectoryTargetBuilder(TrajectorySampling(num_poses=10, time_horizon=5.0))

        logging.basicConfig(level=logging.INFO)

        cache = FeatureCachingPreprocessor(cache_dir=self.cache_dir,
                                           feature_builders=[raster_feature_builder, vectormap_builder],
                                           target_builders=[ego_trajectory_target_builder],
                                           force_feature_computation=False)

        # Extract scenarios
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

        self.compute_features_and_check_builders(scenarios[0], cache, 2, 1)

    def compute_features_and_check_builders(self,
                                            sample: Any,
                                            cache: FeatureCachingPreprocessor,
                                            number_of_features: int,
                                            number_of_targets: int) -> None:
        features, targets = cache.compute_features(sample)
        self.assertEqual(len(targets), number_of_targets)
        self.assertEqual(len(features), number_of_features)

        # Validate Features
        for feature in cache.feature_builders:
            # Check Raster
            self.assertTrue(feature.get_feature_unique_name() in features.keys())
            self.assertIsInstance(features[feature.get_feature_unique_name()], feature.get_feature_type())

        # Validate Targets
        for target in cache.target_builders:
            # Check Raster
            self.assertTrue(target.get_feature_unique_name() in targets.keys())
            self.assertIsInstance(targets[target.get_feature_unique_name()], target.get_feature_type())


if __name__ == '__main__':
    unittest.main()
