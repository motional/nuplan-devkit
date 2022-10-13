import unittest

import torch

from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.test.dummy_vectormap_builder import DummyVectorMapFeature


class TestCollate(unittest.TestCase):
    """Test feature collation functionality."""

    def test_list_as_batch(self) -> None:
        """
        Test collating lists
        """
        single_feature1: FeaturesType = {
            "VectorMap": DummyVectorMapFeature(
                data1=[torch.zeros((13, 3))], data2=[torch.zeros((13, 3))], data3=[{"test": torch.zeros((13, 3))}]
            ),
        }
        single_targets1: FeaturesType = {
            "Trajectory": Trajectory(data=torch.zeros((12, 3))),
        }
        singe_scenario: ScenarioListType = [CachedScenario(log_name='', token='', scenario_type='')]

        to_be_batched = [
            (single_feature1, single_targets1, singe_scenario),
            (single_feature1, single_targets1, singe_scenario),
        ]

        collate = FeatureCollate()
        features, targets, scenarios = collate(to_be_batched)

        vector_map: DummyVectorMapFeature = features["VectorMap"]
        self.assertEqual(vector_map.num_of_batches, 2)
        self.assertEqual(len(vector_map.data1), 2)
        self.assertEqual(vector_map.data1[0].shape, (13, 3))

        trajectory: Trajectory = targets["Trajectory"]
        self.assertEqual(trajectory.data.shape, (2, 12, 3))

        self.assertEqual(len(scenarios), 2)

    def test_collate(self) -> None:
        """
        Test collating features
        """
        single_feature1: FeaturesType = {
            "Trajectory": Trajectory(data=torch.zeros((12, 3))),
            "Raster": Raster(data=torch.zeros((244, 244, 3))),
            "DummyVectorMapFeature": DummyVectorMapFeature(
                data1=[torch.zeros((13, 3))], data2=[torch.zeros((13, 3))], data3=[{"test": torch.zeros((13, 3))}]
            ),
        }
        single_targets1: FeaturesType = {
            "Trajectory": Trajectory(data=torch.zeros((12, 3))),
            "Trajectory2": Trajectory(data=torch.zeros((12, 3))),
        }

        single_feature2: FeaturesType = {
            "Trajectory": Trajectory(data=torch.zeros((12, 3))),
            "Raster": Raster(data=torch.zeros((244, 244, 3))),
            "DummyVectorMapFeature": DummyVectorMapFeature(
                data1=[torch.zeros((13, 3))], data2=[torch.zeros((13, 3))], data3=[{"test": torch.zeros((13, 3))}]
            ),
        }
        single_targets2: FeaturesType = {
            "Trajectory": Trajectory(data=torch.zeros((12, 3))),
            "Trajectory2": Trajectory(data=torch.zeros((12, 3))),
        }

        single_feature3: FeaturesType = {
            "Trajectory": Trajectory(data=torch.zeros((12, 3))),
            "Raster": Raster(data=torch.zeros((244, 244, 3))),
            "DummyVectorMapFeature": DummyVectorMapFeature(
                data1=[torch.zeros((13, 3))], data2=[torch.zeros((13, 3))], data3=[{"test": torch.zeros((13, 3))}]
            ),
        }
        single_targets3: FeaturesType = {
            "Trajectory": Trajectory(data=torch.zeros((12, 3))),
            "Trajectory2": Trajectory(data=torch.zeros((12, 3))),
        }
        singe_scenario: ScenarioListType = [CachedScenario(log_name='', token='', scenario_type='')]

        to_be_batched = [
            (single_feature1, single_targets1, singe_scenario),
            (single_feature2, single_targets2, singe_scenario),
            (single_feature3, single_targets3, singe_scenario),
        ]

        collate = FeatureCollate()
        features, targets, scenarios = collate(to_be_batched)

        self.assertEqual(features["Trajectory"].data.shape, (3, 12, 3))
        self.assertEqual(features["Raster"].data.shape, (3, 244, 244, 3))
        self.assertEqual(features["DummyVectorMapFeature"].num_of_batches, 3)
        self.assertEqual(targets["Trajectory"].data.shape, (3, 12, 3))
        self.assertEqual(len(scenarios), 3)


if __name__ == '__main__':
    unittest.main()
