import unittest

from hydra import compose, initialize_config_dir
from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.utils import SkeletonTestTrain


class TestTrainRasterModel(SkeletonTestTrain):
    """
    Test experiments: raster_model
    """

    def test_open_loop_training_raster_model(self) -> None:
        """
        Tests raster model training in open loop.
        """

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME,
                          overrides=[self.search_path,
                                     *self.default_overrides,
                                     'py_func=train',
                                     '+training=training_raster_model',
                                     'scenario_builder=nuplan_mini',
                                     'splitter=nuplan',
                                     'lightning.trainer.params.max_epochs=1'])
            main(cfg)


if __name__ == '__main__':
    unittest.main()
