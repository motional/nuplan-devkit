import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from hydra import compose, initialize_config_dir

from nuplan.planning.script.builders.scenario_builder import get_local_scenario_cache
from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain
from nuplan.planning.training.preprocessing.feature_builders.test.mock_feature_utils import MockFeatureBuilder


class TestCache(SkeletonTestTrain):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def setUp(self) -> None:
        """
        Set up test attributes.
        """
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.cache_path = f'{self.tmp_dir.name}/cache'
        self.test_args = [
            '+training=training_raster_model',
            'scenario_builder=mock_abstract_scenario_builder',
            f'group={self.tmp_dir.name}',
            f'cache.cache_path={self.cache_path}',
        ]

    def tearDown(self) -> None:
        """
        Cleanup after each test.
        """
        self.tmp_dir.cleanup()

    @patch("nuplan.planning.training.modeling.models.raster_model.RasterModel.get_list_of_required_feature")
    @patch("nuplan.planning.training.modeling.models.raster_model.RasterModel.get_list_of_computed_target")
    def test_cache_dataset(self, feature_builders_fn: Mock, target_builders_fn: Mock) -> None:
        """
        Tests dataset caching.
        """
        feature_builders_fn.return_value = [MockFeatureBuilder(torch.Tensor([0.0]))]
        target_builders_fn.return_value = [MockFeatureBuilder(torch.Tensor([0.0]))]

        # Create cache
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    *self.test_args,
                    'py_func=cache',
                ],
            )
            main(cfg)

        # Check that we can find computed features paths locally
        all_feature_builders = feature_builders_fn.return_value + target_builders_fn.return_value
        all_feature_names = {builder.get_feature_unique_name() for builder in all_feature_builders}
        scenario_cache_paths = get_local_scenario_cache(self.cache_path, all_feature_names)

        # Each scenario stores all it's features together in one of the paths
        self.assertTrue(len(scenario_cache_paths) == cfg.scenario_builder.num_scenarios)


if __name__ == '__main__':
    unittest.main()
