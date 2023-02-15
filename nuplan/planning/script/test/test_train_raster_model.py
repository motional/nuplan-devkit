import unittest

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain


class TestTrainRasterModel(SkeletonTestTrain):
    """
    Test experiments: raster_model
    """

    def test_open_loop_training_raster_model(self) -> None:
        """
        Tests raster model training in open loop.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'py_func=train',
                    '+training=training_raster_model',
                    'scenario_builder=nuplan_mini',
                    'scenario_filter.limit_total_scenarios=16',
                    'splitter=nuplan',
                    'model.model_name=resnet18',
                    'model.pretrained=false',
                    'model.feature_builders.0.target_width=64',
                    'model.feature_builders.0.target_height=64',
                    'lightning.trainer.params.max_epochs=1',
                    'gpu=false',
                ],
            )
            main(cfg)


if __name__ == '__main__':
    unittest.main()
