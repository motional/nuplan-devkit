import unittest

import torch.utils.data

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

NUM_BATCHES = 20


class TestCollateDataLoader(unittest.TestCase):
    """
    Tests data loading functionality
    """

    def setUp(self) -> None:
        """Set up the test case."""
        self.batch_size = 4

        feature_preprocessor = FeaturePreprocessor(
            cache_path=None,
            feature_builders=[
                RasterFeatureBuilder(
                    map_features={'LANE': 1.0, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5},
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
                VectorMapFeatureBuilder(radius=20),
            ],
            target_builders=[EgoTrajectoryTargetBuilder(TrajectorySampling(time_horizon=6.0, num_poses=12))],
            force_feature_computation=False,
        )

        # Keep only a few scenarios instead of testing the whole extraction
        scenario = get_test_nuplan_scenario()
        scenarios = [scenario] * 3

        dataset = ScenarioDataset(scenarios=scenarios, feature_preprocessor=feature_preprocessor)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=False,
            drop_last=True,
            collate_fn=FeatureCollate(),
        )

    def test_dataloader(self) -> None:
        """
        Tests that the training dataloader can be iterated without errors
        """
        dataloader = self.dataloader
        dataloader_iter = iter(dataloader)
        iterations = min(len(dataloader), NUM_BATCHES)

        for _ in range(iterations):
            features, targets, scenarios = next(dataloader_iter)
            self.assertTrue("vector_map" in features.keys())
            vector_map: VectorMap = features["vector_map"]
            self.assertEqual(vector_map.num_of_batches, self.batch_size)
            self.assertEqual(len(vector_map.coords), self.batch_size)
            self.assertEqual(len(vector_map.multi_scale_connections), self.batch_size)


if __name__ == '__main__':
    unittest.main()
