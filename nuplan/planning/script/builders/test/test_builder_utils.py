import math
import os
import unittest
from unittest.mock import Mock, patch

from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_config import (
    update_distributed_lr_scheduler_config,
    update_distributed_optimizer_config,
)


class TestUpdateDistributedTrainingCfg(unittest.TestCase):
    """Test update_distributed_optimizer_config function."""

    world_size = 4

    def setUp(self) -> None:
        """Setup test attributes."""
        self.lr = 1e-5
        self.num_train_batches = 12
        self.batch_size = 2
        self.div_factor = 2
        self.max_lr = 1e-2
        self.betas = [0.9, 0.999]
        self.max_epochs = 2

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_optimizer_config_equal_variance(self) -> None:
        """Test default setting where the lr is scaled to maintain equal variance."""
        cfg_mock = Mock(DictConfig)

        # Mock optimizer config
        optimizer = Mock()
        optimizer._target_ = 'torch.optim.Adam'
        optimizer.betas = self.betas.copy()
        optimizer.lr = self.lr

        # Mock lightning config
        lightning = Mock()
        lightning.distributed_training.equal_variance_scaling_strategy = True

        cfg_mock.optimizer = optimizer
        cfg_mock.lightning = lightning

        # test method of scaling lr to maintain equal_variance
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        msg = f'Expected {(self.world_size**0.5)*self.lr} but got {cfg_mock.optimizer.lr}'
        msg_beta_1 = f'Expected {self.betas[0]}, {self.world_size ** 0.5}, {self.betas[0]**(self.world_size ** 0.5)} but got {cfg_mock.optimizer.betas[0]}'
        msg_beta_2 = f'Expected {self.betas[1]**(self.world_size ** 0.5)} but got {cfg_mock.optimizer.betas[1]}'

        self.assertTrue(
            float(cfg_mock.optimizer.lr) == (self.world_size**0.5) * self.lr,
            msg=msg,
        )
        self.assertTrue(
            float(cfg_mock.optimizer.betas[0]) == self.betas[0] ** (self.world_size**0.5),
            msg=msg_beta_1,
        )
        self.assertTrue(
            float(cfg_mock.optimizer.betas[1]) == self.betas[1] ** (self.world_size**0.5),
            msg=msg_beta_2,
        )

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_optimizer_config_linearly(self) -> None:
        """Test default setting where the lr is scaled linearly."""
        cfg_mock = Mock(DictConfig)

        # Mock optimizer config
        optimizer = Mock()
        optimizer._target_ = 'torch.optim.Adam'
        optimizer.betas = self.betas.copy()
        optimizer.lr = self.lr

        # Mock lightning config
        lightning = Mock()
        lightning.distributed_training.equal_variance_scaling_strategy = False

        cfg_mock.optimizer = optimizer
        cfg_mock.lightning = lightning

        # test method of scaling lr linearly
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        msg = f'Expected {self.world_size*self.lr} but got {cfg_mock.optimizer.lr}'
        msg_beta_1 = f'Expected {self.betas[0]**self.world_size} but got {cfg_mock.optimizer.betas[0]}'
        msg_beta_2 = f'Expected {self.betas[1]**self.world_size} but got {cfg_mock.optimizer.betas[1]}'
        print((self.betas[0] ** (self.world_size)), self.betas[0], self.world_size)
        self.assertTrue(float(cfg_mock.optimizer.lr) == self.world_size * self.lr, msg=msg)
        self.assertTrue(
            float(cfg_mock.optimizer.betas[0]) == (self.betas[0] ** (self.world_size)),
            msg=msg_beta_1,
        )
        self.assertTrue(
            float(cfg_mock.optimizer.betas[1]) == (self.betas[1] ** (self.world_size)),
            msg=msg_beta_2,
        )

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_not_one_cycle_lr(self) -> None:
        """
        Test default setting where the lr_scheduler is not supported.
        Currently, anything other than OneCycleLR is not supported.
        """
        cfg_mock = Mock(DictConfig)
        # Mock lr_scheduler using ExponentialLR, which is not supported for scaling wrt multinode setting
        lr_scheduler = Mock()
        lr_scheduler._target_ = 'torch.optim.lr_scheduler.ExponentialLR'
        lr_scheduler.gamma = 0.9
        lr_scheduler.steps_per_epoch = None

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.overfitting.enable = True
        lightning.trainer.overfitting.params.overfit_batches = 1
        lightning.distributed_training.equal_variance_scaling_strategy = False
        lightning.trainer.params.max_epochs = self.max_epochs

        # Mock dataloader config
        data_loader = Mock()
        data_loader.params.batch_size = self.batch_size

        cfg_mock.lr_scheduler = lr_scheduler
        cfg_mock.lightning = lightning
        cfg_mock.data_loader = data_loader

        # test that the steps_per_epoch attribute of the cfg was not edited
        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)
        msg_steps_per_epoch = f'Expected Mock to not be edited, but steps_per_epoch was edited: steps_per_epoch is {cfg_mock.lr_scheduler.steps_per_epoch}'

        self.assertTrue(cfg_mock.lr_scheduler.steps_per_epoch is None, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_oclr_overfit_zero_batches(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 0."""
        cfg_mock = Mock(DictConfig)
        # Mock lr_scheduler config
        lr_scheduler = Mock()
        lr_scheduler._target_ = 'torch.optim.lr_scheduler.OneCycleLR'
        lr_scheduler.max_lr = self.max_lr
        lr_scheduler.steps_per_epoch = None
        lr_scheduler.div_factor = self.div_factor

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.overfitting.enable = True
        lightning.trainer.overfitting.params.overfit_batches = 0
        lightning.distributed_training.equal_variance_scaling_strategy = False
        lightning.trainer.params.max_epochs = self.max_epochs

        # Mock dataloader config
        data_loader = Mock()
        data_loader.params.batch_size = self.batch_size

        cfg_mock.lr_scheduler = lr_scheduler
        cfg_mock.lightning = lightning
        cfg_mock.data_loader = data_loader

        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)

        expected_steps_per_epoch = math.ceil(math.ceil(self.num_train_batches / self.world_size) / self.max_epochs)

        msg_steps_per_epoch = (
            f'Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}'
        )
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_overfit_one_batches(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 1."""
        cfg_mock = Mock(DictConfig)
        # Mock lr_scheduler config
        lr_scheduler = Mock()
        lr_scheduler._target_ = 'torch.optim.lr_scheduler.OneCycleLR'
        lr_scheduler.max_lr = self.max_lr
        lr_scheduler.steps_per_epoch = None
        lr_scheduler.div_factor = self.div_factor

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.overfitting.enable = True
        lightning.trainer.overfitting.params.overfit_batches = 1
        lightning.distributed_training.equal_variance_scaling_strategy = False
        lightning.trainer.params.max_epochs = self.max_epochs

        # Mock dataloader config
        data_loader = Mock()
        data_loader.params.batch_size = self.batch_size

        cfg_mock.lr_scheduler = lr_scheduler
        cfg_mock.lightning = lightning
        cfg_mock.data_loader = data_loader

        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)

        expected_steps_per_epoch = math.ceil(
            lightning.trainer.overfitting.params.overfit_batches / self.world_size / self.max_epochs
        )

        msg_steps_per_epoch = (
            f'Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}'
        )

        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_overfit_batches_fractional(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 1."""
        cfg_mock = Mock(DictConfig)
        # Mock lr_scheduler config
        lr_scheduler = Mock()
        lr_scheduler._target_ = 'torch.optim.lr_scheduler.OneCycleLR'
        lr_scheduler.max_lr = self.max_lr
        lr_scheduler.steps_per_epoch = None
        lr_scheduler.div_factor = self.div_factor

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.overfitting.enable = True
        lightning.trainer.overfitting.params.overfit_batches = 0.5
        lightning.distributed_training.equal_variance_scaling_strategy = False
        lightning.trainer.params.max_epochs = self.max_epochs

        # Mock dataloader config
        data_loader = Mock()
        data_loader.params.batch_size = self.batch_size

        cfg_mock.lr_scheduler = lr_scheduler
        cfg_mock.lightning = lightning
        cfg_mock.data_loader = data_loader

        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)

        batches_to_overfit = math.ceil(self.num_train_batches * lightning.trainer.overfitting.params.overfit_batches)
        expected_steps_per_epoch = math.ceil(math.ceil(batches_to_overfit / self.world_size) / self.max_epochs)

        msg_steps_per_epoch = (
            f'Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}'
        )
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)


if __name__ == '__main__':
    unittest.main()
