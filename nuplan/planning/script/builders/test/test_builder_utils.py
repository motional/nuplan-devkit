import math
import os
import unittest
from unittest.mock import Mock, patch

from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_config import (
    update_distributed_lr_scheduler_config,
    update_distributed_optimizer_config,
)

TEST_WORLD_SIZE = '4'
TEST_LR = 1e-5
TEST_TRAIN_DATASET_LENGTH = 12
TEST_BATCH_SIZE = 2
TEST_DIV_FACTOR = 2
TEST_MAX_LR = 1e-2


class TestUpdateDistributedTrainingCfg(unittest.TestCase):
    """Test update_distributed_optimizer_config function."""

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_optimizer_config_no_scaling(self) -> None:
        """Test default setting where the lr is not scaled"""
        cfg_mock = Mock(DictConfig)

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = 'ddp'
        lightning.distributed_training.lr_scaling_method = 'none'

        cfg_mock.optimizer = optimizer
        cfg_mock.lightning = lightning

        # test default setting where lr is not scaled
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        msg = f'Learning rate was not supposed to be scaled. Expected {TEST_LR} but got {cfg_mock.optimizer.lr}'
        self.assertTrue(float(cfg_mock.optimizer.lr) == float(TEST_LR), msg=msg)

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_optimizer_config_not_ddp(self) -> None:
        """Test default setting where the training strategy is not ddp, in which case lr should not be scaled"""
        cfg_mock = Mock(DictConfig)

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = (
            'nonexistent_strategy'  # dp splits batchsize across k gpus, so effective batch size is the same
        )
        lightning.distributed_training.lr_scaling_method = 'linearly'

        cfg_mock.optimizer = optimizer
        cfg_mock.lightning = lightning

        # test default setting where lr is not scaled
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        msg = f'Learning rate was not supposed to be scaled. Expected {TEST_LR} but got {cfg_mock.optimizer.lr}'
        self.assertTrue(float(cfg_mock.optimizer.lr) == float(TEST_LR), msg=msg)

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_optimizer_config_equal_variance(self) -> None:
        """Test default setting where the lr is scaled to maintain equal variance"""
        cfg_mock = Mock(DictConfig)

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = 'ddp'
        lightning.distributed_training.lr_scaling_method = 'equal_variance'

        cfg_mock.optimizer = optimizer
        cfg_mock.lightning = lightning

        # test method of scaling lr to maintain equal_variance
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        msg = f'Expected {(float(TEST_WORLD_SIZE)**0.5)*TEST_LR} but got {cfg_mock.optimizer.lr}'
        self.assertTrue(
            float(cfg_mock.optimizer.lr) == (float(TEST_WORLD_SIZE) ** 0.5) * TEST_LR,
            msg=msg,
        )

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_optimizer_config_linearly(self) -> None:
        """Test default setting where the lr is scaled linearly"""
        cfg_mock = Mock(DictConfig)

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = 'ddp'
        lightning.distributed_training.lr_scaling_method = 'linearly'

        cfg_mock.optimizer = optimizer
        cfg_mock.lightning = lightning

        # test method of scaling lr linearly
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        msg = f'Expected {float(TEST_WORLD_SIZE)*TEST_LR} but got {cfg_mock.optimizer.lr}'
        self.assertTrue(float(cfg_mock.optimizer.lr) == float(TEST_WORLD_SIZE) * TEST_LR, msg=msg)

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_optimizer_config_not_supported(self) -> None:
        """Test default setting where the scaling method is not supported"""
        cfg_mock = Mock(DictConfig)

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = 'ddp'
        lightning.distributed_training.lr_scaling_method = 'nonexistent_method'

        cfg_mock.optimizer = optimizer
        cfg_mock.lightning = lightning

        # test scaling method that is not supported
        with self.assertRaises(RuntimeError):
            cfg_mock = update_distributed_optimizer_config(cfg_mock)

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_lr_scheduler_config_not_one_cycle_lr(self) -> None:
        """
        Test default setting where the lr_scheduler is not supported.
        Currently, anything other than OneCycleLR is not supported
        """
        cfg_mock = Mock(DictConfig)
        # Mock lr_scheduler using ExponentialLR, which is not supported for scaling wrt multinode setting
        lr_scheduler = Mock()
        lr_scheduler._target_ = 'torch.optim.lr_scheduler.ExponentialLR'
        lr_scheduler.gamma = 0.9
        lr_scheduler.steps_per_epoch = None

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = 'ddp'
        lightning.trainer.overfitting.params.overfit_batches = 1
        lightning.distributed_training.lr_scaling_method = 'linearly'

        # Mock dataloader config
        data_loader = Mock()
        data_loader.params.batch_size = TEST_BATCH_SIZE

        cfg_mock.optimizer = optimizer
        cfg_mock.lr_scheduler = lr_scheduler
        cfg_mock.lightning = lightning
        cfg_mock.data_loader = data_loader
        cfg_mock.optimizer = optimizer

        # update optimizer, because optimizer.lr will be used for computing max_lr
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        # test that the steps_per_epoch attribute of the cfg was not edited
        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, train_dataset_len=TEST_TRAIN_DATASET_LENGTH)
        msg_steps_per_epoch = 'Expected Mock to not be edited, but steps_per_epoch was edited'

        self.assertTrue(cfg_mock.lr_scheduler.steps_per_epoch is None, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_lr_scheduler_config_not_ddp(self) -> None:
        """
        Test default setting where the lr_scheduler is not supported.
        Currently, this is only supported for ddp.
        """
        cfg_mock = Mock(DictConfig)
        # Mock lr_scheduler config
        lr_scheduler = Mock()
        lr_scheduler._target_ = 'torch.optim.lr_scheduler.OneCycleLR'
        lr_scheduler.max_lr = TEST_MAX_LR
        lr_scheduler.steps_per_epoch = None
        lr_scheduler.div_factor = TEST_DIV_FACTOR

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = (
            'nonexistent_strategy'  # currently strategies other than ddp are not supported
        )
        lightning.trainer.overfitting.params.overfit_batches = 1
        lightning.distributed_training.lr_scaling_method = 'linearly'

        # Mock dataloader config
        data_loader = Mock()
        data_loader.params.batch_size = TEST_BATCH_SIZE

        cfg_mock.optimizer = optimizer
        cfg_mock.lr_scheduler = lr_scheduler
        cfg_mock.lightning = lightning
        cfg_mock.data_loader = data_loader
        cfg_mock.optimizer = optimizer

        # update optimizer, because optimizer.lr will be used for computing max_lr
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        # test that the steps_per_epoch attribute of the cfg was not edited
        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, train_dataset_len=TEST_TRAIN_DATASET_LENGTH)
        msg_steps_per_epoch = 'Expected Mock to not be edited, but steps_per_epoch was edited'

        self.assertTrue(cfg_mock.lr_scheduler.steps_per_epoch is None, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_lr_scheduler_config_oclr_overfit_zero_batches(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 0"""
        cfg_mock = Mock(DictConfig)
        # Mock lr_scheduler config
        lr_scheduler = Mock()
        lr_scheduler._target_ = 'torch.optim.lr_scheduler.OneCycleLR'
        lr_scheduler.max_lr = TEST_MAX_LR
        lr_scheduler.steps_per_epoch = None
        lr_scheduler.div_factor = TEST_DIV_FACTOR

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = 'ddp'
        lightning.trainer.overfitting.params.overfit_batches = 0
        lightning.distributed_training.lr_scaling_method = 'linearly'

        # Mock dataloader config
        data_loader = Mock()
        data_loader.params.batch_size = TEST_BATCH_SIZE

        cfg_mock.optimizer = optimizer
        cfg_mock.lr_scheduler = lr_scheduler
        cfg_mock.lightning = lightning
        cfg_mock.data_loader = data_loader
        cfg_mock.optimizer = optimizer

        # update optimizer, because optimizer.lr will be used for computing max_lr
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, train_dataset_len=TEST_TRAIN_DATASET_LENGTH)

        expected_steps_per_epoch = math.ceil(
            math.ceil(TEST_TRAIN_DATASET_LENGTH / int(TEST_WORLD_SIZE)) / TEST_BATCH_SIZE
        )
        expected_max_lr = cfg_mock.optimizer.lr * TEST_DIV_FACTOR

        msg_steps_per_epoch = (
            f'Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}'
        )
        msg_max_lr = f'Expected steps per epoch to be {expected_max_lr} but got {cfg_mock.lr_scheduler.max_lr}'
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)
        self.assertEqual(cfg_mock.lr_scheduler.max_lr, expected_max_lr, msg=msg_max_lr)

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_lr_scheduler_config_overfit_one_batches(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 1"""
        cfg_mock = Mock(DictConfig)
        # Mock lr_scheduler config
        lr_scheduler = Mock()
        lr_scheduler._target_ = 'torch.optim.lr_scheduler.OneCycleLR'
        lr_scheduler.max_lr = TEST_MAX_LR
        lr_scheduler.steps_per_epoch = None
        lr_scheduler.div_factor = TEST_DIV_FACTOR

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = 'ddp'
        lightning.trainer.overfitting.params.overfit_batches = 1
        lightning.distributed_training.lr_scaling_method = 'linearly'

        # Mock dataloader config
        data_loader = Mock()
        data_loader.params.batch_size = TEST_BATCH_SIZE

        cfg_mock.optimizer = optimizer
        cfg_mock.lr_scheduler = lr_scheduler
        cfg_mock.lightning = lightning
        cfg_mock.data_loader = data_loader
        cfg_mock.optimizer = optimizer

        # update optimizer, because optimizer.lr will be used for computing max_lr
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, train_dataset_len=TEST_TRAIN_DATASET_LENGTH)

        expected_steps_per_epoch = math.ceil(
            lightning.trainer.overfitting.params.overfit_batches / int(TEST_WORLD_SIZE)
        )
        expected_max_lr = cfg_mock.optimizer.lr * TEST_DIV_FACTOR

        msg_steps_per_epoch = (
            f'Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}'
        )
        msg_max_lr = f'Expected steps per epoch to be {expected_max_lr} but got {cfg_mock.lr_scheduler.max_lr}'
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)
        self.assertEqual(cfg_mock.lr_scheduler.max_lr, expected_max_lr, msg=msg_max_lr)

    @patch.dict(os.environ, {"WORLD_SIZE": TEST_WORLD_SIZE}, clear=True)
    def test_update_distributed_lr_scheduler_config_overfit_batches_fractional(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 1"""
        cfg_mock = Mock(DictConfig)
        # Mock lr_scheduler config
        lr_scheduler = Mock()
        lr_scheduler._target_ = 'torch.optim.lr_scheduler.OneCycleLR'
        lr_scheduler.max_lr = TEST_MAX_LR
        lr_scheduler.steps_per_epoch = None
        lr_scheduler.div_factor = TEST_DIV_FACTOR

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock optimizer config
        optimizer = Mock()
        optimizer.lr = TEST_LR

        # Mock lightning config
        lightning = Mock()
        lightning.trainer.params.accelerator = 'ddp'
        lightning.trainer.overfitting.params.overfit_batches = 0.5
        lightning.distributed_training.lr_scaling_method = 'linearly'

        # Mock dataloader config
        data_loader = Mock()
        data_loader.params.batch_size = TEST_BATCH_SIZE

        cfg_mock.optimizer = optimizer
        cfg_mock.lr_scheduler = lr_scheduler
        cfg_mock.lightning = lightning
        cfg_mock.data_loader = data_loader
        cfg_mock.optimizer = optimizer

        # update optimizer, because optimizer.lr will be used for computing max_lr
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, train_dataset_len=TEST_TRAIN_DATASET_LENGTH)

        batches_to_overfit = math.ceil(TEST_TRAIN_DATASET_LENGTH * lightning.trainer.overfitting.params.overfit_batches)
        expected_steps_per_epoch = math.ceil(math.ceil(batches_to_overfit / int(TEST_WORLD_SIZE)) / TEST_BATCH_SIZE)
        expected_max_lr = cfg_mock.optimizer.lr * TEST_DIV_FACTOR

        msg_steps_per_epoch = (
            f'Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}'
        )
        msg_max_lr = f'Expected steps per epoch to be {expected_max_lr} but got {cfg_mock.lr_scheduler.max_lr}'
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)
        self.assertEqual(cfg_mock.lr_scheduler.max_lr, expected_max_lr, msg=msg_max_lr)


if __name__ == '__main__':
    unittest.main()
