import unittest

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain


class TestTrainUrbanDriverOpenLoopModel(SkeletonTestTrain):
    """
    Test experiments: urban_driver_open_loop_model
    """

    def test_open_loop_training_urban_driver_open_loop_model(self) -> None:
        """
        Tests urban_driver_open_loop model training in open loop.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'py_func=train',
                    '+training=training_urban_driver_open_loop_model',
                    'scenario_builder=nuplan_mini',
                    'scenario_filter.limit_total_scenarios=32',
                    'splitter=nuplan',
                    'lightning.trainer.params.max_epochs=1',
                    'cache.force_feature_computation=True',
                ],
            )
            main(cfg)


if __name__ == '__main__':
    unittest.main()
