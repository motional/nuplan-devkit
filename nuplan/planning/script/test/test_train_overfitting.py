import unittest

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain


class TestTrain(SkeletonTestTrain):
    """
    Test main training entry point using combinations of models, datasets, filters etc.
    """

    def test_raster_model_overfitting(self) -> None:
        """
        Tests raster model overfitting in open loop.
        """
        loss_threshold = 2.0  # loss threshold that should be reached for the model to be considered overfit

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'log_config=false',
                    'py_func=train',
                    '+training=training_raster_model',
                    'scenario_builder=nuplan_mini',
                    'scenario_filter.limit_total_scenarios=15',
                    'splitter=nuplan',
                    'optimizer.lr=0.01',
                    'lightning.trainer.overfitting.enable=true',
                    'lightning.trainer.overfitting.params.max_epochs=200',
                    'data_loader.params.batch_size=2',
                    'data_loader.params.num_workers=2',
                ],
            )
            engine = main(cfg)
            self.assertLessEqual(engine.trainer.callback_metrics['loss/train_loss'], loss_threshold)

    def test_urban_driver_open_loop_model_overfitting(self) -> None:
        """
        Tests urban_driver_open_loop model overfitting in open loop.
        """
        loss_threshold = 2.0  # loss threshold that should be reached for the model to be considered overfit

        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'log_config=false',
                    'py_func=train',
                    '+training=training_urban_driver_open_loop_model',
                    'data_augmentation=[]',
                    'scenario_builder=nuplan_mini',
                    'scenario_filter.limit_total_scenarios=15',
                    'splitter=nuplan',
                    'optimizer=adamw',
                    'optimizer.lr=1.25e-5',
                    'lightning.trainer.overfitting.enable=true',
                    'lightning.trainer.overfitting.params.max_epochs=300',
                    'data_loader.params.batch_size=1',
                    'data_loader.params.num_workers=2',
                ],
            )
            engine = main(cfg)
            self.assertLessEqual(engine.trainer.callback_metrics['loss/train_loss'], loss_threshold)


if __name__ == '__main__':
    unittest.main()
