import unittest

from hydra import compose, initialize_config_dir
from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.utils import SkeletonTestTrain


class TestTrain(SkeletonTestTrain):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def test_raster_model_overfitting(self) -> None:
        """
        Tests raster model overfitting in open loop.
        """

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME,
                          overrides=[self.search_path,
                                     *self.default_overrides,
                                     'log_config=false',
                                     'py_func=train',
                                     '+training=training_raster_model',
                                     'scenario_builder=nuplan_mini',
                                     'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=10',
                                     'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.1',
                                     'splitter=nuplan',
                                     'lightning.optimization.optimizer.learning_rate=0.1',
                                     'lightning.trainer.overfitting.enable=true',
                                     'lightning.trainer.overfitting.params.max_epochs=200',
                                     'data_loader.params.batch_size=2',
                                     'data_loader.params.num_workers=2'])
            engine = main(cfg)
            self.assertLessEqual(engine.trainer.callback_metrics['loss/train_loss'], 1.0)


if __name__ == '__main__':
    unittest.main()
