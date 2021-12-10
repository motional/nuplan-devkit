import unittest

from hydra import compose, initialize_config_dir
from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.utils import SkeletonTestTrain


class TestCache(SkeletonTestTrain):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    @unittest.skip('Skip due to log-split mismatch')
    def test_cache_dataset(self) -> None:
        """
        Tests dataset caching.
        """

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME,
                          overrides=[self.search_path,
                                     *self.default_overrides,
                                     'py_func=cache_data',
                                     '+training=training_raster_model',
                                     'scenario_builder=nuplan_mini',
                                     'splitter=nuplan'])
            main(cfg)


if __name__ == '__main__':
    unittest.main()
