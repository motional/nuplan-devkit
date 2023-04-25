import unittest

import numpy as np
import pytorch_lightning as pl
import ray
from omegaconf import DictConfig

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario_builder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.data_augmentation.kinematic_agent_augmentation import KinematicAgentAugmentor
from nuplan.planning.training.data_loader.datamodule import DataModule
from nuplan.planning.training.data_loader.log_splitter import LogSplitter
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.training.preprocessing.test.dummy_vectormap_builder import DummyVectorMapBuilder
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class SkeletonTestDataloader(unittest.TestCase):
    """
    Skeleton with initialized dataloader used in testing.
    """

    def setUp(self) -> None:
        """
        Set up basic configs.
        """
        pl.seed_everything(2022, workers=True)

        # Create splitter
        self.splitter = LogSplitter(
            log_splits={
                'train': ["2021.07.16.20.45.29_veh-35_01095_01486"],
                'val': ["2021.06.07.18.53.26_veh-26_00005_00427"],
                'test': ["2021.10.06.07.26.10_veh-52_00006_00398"],
            }
        )

        # Create feature builder
        feature_builders = [
            DummyVectorMapBuilder(),
            VectorMapFeatureBuilder(radius=20),
            AgentsFeatureBuilder(TrajectorySampling(num_poses=4, time_horizon=1.5)),
            RasterFeatureBuilder(
                map_features={'LANE': 1, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5},
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
        ]
        target_builders = [EgoTrajectoryTargetBuilder(TrajectorySampling(num_poses=10, time_horizon=5.0))]
        self.feature_preprocessor = FeaturePreprocessor(
            cache_path=None,
            force_feature_computation=True,
            feature_builders=feature_builders,
            target_builders=target_builders,
        )

        # Extract scenarios
        self.scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=150,
            expand_scenarios=True,
            remove_invalid_goals=False,
            shuffle=True,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            ego_start_speed_threshold=None,
            ego_stop_speed_threshold=None,
            speed_noise_tolerance=None,
            token_set_path=None,
            fraction_in_token_set_threshold=None,
        )

        self.augmentors = [
            KinematicAgentAugmentor(
                trajectory_length=10,
                dt=0.1,
                mean=[0.3, 0.1, np.pi / 12],
                std=[0.5, 0.1, np.pi / 12],
                low=[-0.2, 0.0, 0.0],
                high=[0.8, 0.2, np.pi / 6],
                augment_prob=0.5,
            )
        ]
        self.scenario_builder = get_test_nuplan_scenario_builder()

    def _test_dataloader(self, worker: WorkerPool) -> None:
        """
        Tests that the training dataloader can be iterated without errors
        """
        scenarios = self.scenario_builder.get_scenarios(self.scenario_filter, worker)
        self.assertGreater(len(scenarios), 0)

        # Construct data module
        batch_size = 4
        num_workers = 4
        scenario_type_sampling_weights = DictConfig({'enable': False, 'scenario_type_weights': {'unknown': 1.0}})

        datamodule = DataModule(
            feature_preprocessor=self.feature_preprocessor,
            splitter=self.splitter,
            train_fraction=1.0,
            val_fraction=0.1,
            test_fraction=0.1,
            all_scenarios=scenarios,
            augmentors=self.augmentors,
            worker=worker,
            scenario_type_sampling_weights=scenario_type_sampling_weights,
            dataloader_params={"batch_size": batch_size, "num_workers": num_workers, "drop_last": True},
        )

        # Initialize data module
        datamodule.setup('fit')

        self.assertGreater(len(datamodule.train_dataloader()), 0)

        # Run
        for features, targets, scenarios in datamodule.train_dataloader():
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

    def tearDown(self) -> None:
        """
        Clean up.
        """
        if ray.is_initialized():
            ray.shutdown()


if __name__ == '__main__':
    unittest.main()
