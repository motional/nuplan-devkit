import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain


class TestCache(SkeletonTestTrain):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def test_cache_dataset(self) -> None:
        """
        Tests dataset caching.
        """
        tmp_dir = tempfile.TemporaryDirectory()
        cache_path = f'{tmp_dir.name}/cache'

        test_args = [
            '+training=training_raster_model',
            'scenario_builder=nuplan_mini',
            'splitter=nuplan',
            f'group={tmp_dir.name}',
            f'cache.cache_path={cache_path}',
        ]

        # Create cache
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    self.search_path,
                    *self.default_overrides,
                    *test_args,
                    'py_func=cache',
                ],
            )
            main(cfg)

        assert any(Path(cache_path).iterdir())

        # Use cache without loading the dataset
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    self.search_path,
                    *self.default_overrides,
                    *test_args,
                    'py_func=train',
                    'cache.cleanup_cache=false',
                    'cache.use_cache_without_dataset=true',
                ],
            )
            main(cfg)

        # Use cache with loading the dataset
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    self.search_path,
                    *self.default_overrides,
                    *test_args,
                    'py_func=train',
                    'cache.cleanup_cache=false',
                    'cache.use_cache_without_dataset=false',
                ],
            )
            main(cfg)


if __name__ == '__main__':
    unittest.main()
