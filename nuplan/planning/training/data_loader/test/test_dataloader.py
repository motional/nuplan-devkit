import os
import unittest

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.data_loader.datamodule import DataModule
from nuplan.planning.training.data_loader.log_splitter import LogSplitter
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_caching_preprocessor import FeatureCachingPreprocessor
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import \
    EgoTrajectoryTargetBuilder
from nuplan.planning.training.preprocessing.test.dummy_vectormap_builder import DummyVectorMapBuilder
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


class TestDataloader(unittest.TestCase):
    """
    Tests data loading functionality
    """

    def _test_dataloader(self, scenario_builder: NuPlanScenarioBuilder, worker: WorkerPool) -> None:
        """
        Tests that the training dataloader can be iterated without errors
        """
        # Create splitter
        splitter = LogSplitter(
            log_splits={
                'train': ["2021.05.26.20.05.14_38_1622073985538950.8_1622074969538793.5"],
                'val': ["2021.06.04.19.10.47_47_1622848319071793.5_1622849413071686.2"],
                'test': ["2021.05.28.21.56.29_24_1622239057169313.0_1622240664170207.2"],
            })

        # Create feature builder
        features = FeatureCachingPreprocessor(cache_dir=None,
                                              force_feature_computation=True,
                                              feature_builders=[
                                                  DummyVectorMapBuilder(),
                                                  VectorMapFeatureBuilder(radius=20),
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
                                                  )],
                                              target_builders=[
                                                  EgoTrajectoryTargetBuilder(TrajectorySampling(
                                                      num_poses=10,
                                                      time_horizon=5.0)
                                                  )])

        # Extract scenarios
        scenario_filter = ScenarioFilters(
            log_names=None,
            log_labels=None,
            max_scenarios_per_log=None,
            scenario_types=None,
            scenario_tokens=None,
            map_name=None,
            shuffle=False,
            limit_scenarios_per_type=100,
            subsample_ratio=0.05,
            flatten_scenarios=True,
            remove_invalid_goals=True,
            limit_total_scenarios=20,
        )

        scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

        self.assertGreater(len(scenarios), 0)

        # Construct data module
        batch_size = 4
        num_workers = 4
        datamodule = DataModule(
            feature_and_targets_builders=features,
            splitter=splitter,
            train_fraction=1.0,
            val_fraction=0.1,
            test_fraction=0.1,
            all_scenarios=scenarios,
            dataloader_params={"batch_size": batch_size, "num_workers": num_workers, "drop_last": True},
        )

        # Initialize data module
        datamodule.setup('fit')

        self.assertGreater(len(datamodule.train_dataloader()), 0)

        # Run
        for features, targets in datamodule.train_dataloader():
            # Validate that all features and targets are preset
            self.assertTrue("raster" in features.keys())
            self.assertTrue("vector_map" in features.keys())
            self.assertTrue("trajectory" in targets.keys())

            # Validate the dimensions
            scenario_features: Raster = features["raster"]
            trajectory_target: Trajectory = targets["trajectory"]
            self.assertEqual(scenario_features.num_batches, trajectory_target.num_batches)
            self.assertIsInstance(scenario_features, Raster)
            self.assertIsInstance(trajectory_target, Trajectory)
            self.assertEqual(scenario_features.num_batches, batch_size)

    def test_dataloader_nuplan_ray(self) -> None:
        """
        Test dataloader using nuPlan DB.
        """
        scenario_builder = NuPlanScenarioBuilder(
            version='nuplan_v0.1_mini',
            data_root=os.getenv('NUPLAN_DATA_ROOT'))
        scenario_builder = NuPlanScenarioBuilder(version='nuplan_v0.1_mini', data_root='/data/sets/nuplan')
        self._test_dataloader(scenario_builder, RayDistributed())

    def test_dataloader_nuplan_sequential(self) -> None:
        """
        Test dataloader using nuPlan DB using a sequential worker.
        """
        scenario_builder = NuPlanScenarioBuilder(
            version='nuplan_v0.1_mini',
            data_root=os.getenv('NUPLAN_DATA_ROOT'))
        scenario_builder = NuPlanScenarioBuilder(version='nuplan_v0.1_mini', data_root='/data/sets/nuplan')
        self._test_dataloader(scenario_builder, Sequential())


if __name__ == '__main__':
    unittest.main()
