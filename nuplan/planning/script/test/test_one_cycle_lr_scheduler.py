import os
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain


class TestTrainOptimizerOCLRScheduler(SkeletonTestTrain):
    """
    Test Optimizer and LR Scheduler instantiation.
    """

    world_size = 4

    def setUp(self) -> None:
        """Setup test attributes."""
        super().setUp()
        self.optimizer_initial_lr = 1e-2
        self.div_factor = 20
        self.max_lr = 2
        self.steps_per_epoch = 20

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=False)
    def test_optimizer_oclr_scheduler_instantiation(self) -> None:
        """
        Tests that optimizer and lr_scheduler were instantiated correctly.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'py_func=train',
                    '+training=training_simple_vector_model',
                    'scenario_builder=nuplan_mini',
                    'scenario_filter.limit_total_scenarios=30',
                    'splitter=nuplan',
                    'lightning.trainer.params.max_epochs=1',
                    'gpu=false',
                    'optimizer=adamw',
                    f'optimizer.lr={str(self.optimizer_initial_lr)}',
                    'lr_scheduler=one_cycle_lr',
                    f'lr_scheduler.div_factor={str(self.div_factor)}',
                    f'lr_scheduler.max_lr={str(self.max_lr)}',
                    f'lr_scheduler.steps_per_epoch={str(self.steps_per_epoch)}',
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
            # Check that base lr = max lr / div_factor = initial lr specified in optimizer / div_factor.
            expected_base_lr = self.optimizer_initial_lr / self.div_factor
            result_base_lr = engine.model.lr_schedulers().state_dict()['base_lrs'][0]
            self.assertEqual(
                result_base_lr,
                expected_base_lr,
                msg=f'Expected base lr to be {expected_base_lr} but got {result_base_lr}',
            )
            self.tearDown()


if __name__ == '__main__':
    unittest.main()
