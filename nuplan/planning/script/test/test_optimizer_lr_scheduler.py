import unittest

import torch
from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain

TEST_OPTIMIZER_INITIAL_LR = 1e-2
TEST_DIV_FACTOR = 20
TEST_MAX_LR = 2
TEST_WORLD_SIZE = '4'
TEST_STEPS_PER_EPOCH = 20
TEST_EPOCHS = 1


class TestTrainOptimizerOCLRScheduler(SkeletonTestTrain):
    """
    Test experiments: raster_model
    """

    def test_optimizer_oclr_scheduler_instantiation(self) -> None:
        """
        Tests that optimizer and lr_scheduler were instantiated correctly.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    self.search_path,
                    *self.default_overrides,
                    'py_func=train',
                    '+training=training_raster_model',
                    'scenario_builder=nuplan_mini',
                    'splitter=nuplan',
                    'model.model_name=resnet18',
                    'model.pretrained=false',
                    'model.feature_builders.0.target_width=64',
                    'model.feature_builders.0.target_height=64',
                    'lightning.trainer.params.max_epochs=1',
                    'gpu=false',
                    'optimizer=adamw',
                    f'optimizer.lr={str(TEST_OPTIMIZER_INITIAL_LR)}',
                    'lr_scheduler=one_cycle_lr',
                    f'lr_scheduler.div_factor={str(TEST_DIV_FACTOR)}',
                    f'lr_scheduler.max_lr={str(TEST_MAX_LR)}',
                    f'lr_scheduler.steps_per_epoch={str(TEST_STEPS_PER_EPOCH)}',
                    f'lr_scheduler.steps_per_epoch={str(TEST_EPOCHS)}',
                ],
            )
            engine = main(cfg)
            self.assertTrue(
                isinstance(engine.model.optimizers(), torch.optim.AdamW),
                msg=f'Expected optimizer {torch.optim.AdamW} but got {engine.model.optimizers()}',
            )
            self.assertTrue(
                isinstance(engine.model.lr_schedulers(), torch.optim.lr_scheduler.OneCycleLR),
                msg=f'Expected lr_scheduler {torch.optim.lr_scheduler.OneCycleLR} but got {engine.model.lr_schedulers()}',
            )
            self.tearDown()


if __name__ == '__main__':
    unittest.main()
