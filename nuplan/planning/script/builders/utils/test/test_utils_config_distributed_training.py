import math
import os
import unittest
from unittest.mock import patch

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
        self.exponential_lr_scheduler_cfg = {
            "_target_": "torch.optim.lr_scheduler.ExponentialLR",
            "gamma": 0.9,
            "steps_per_epoch": None,
        }
        self.one_cycle_lr_scheduler_cfg = {
            "_target_": "torch.optim.lr_scheduler.OneCycleLR",
            "max_lr": self.max_lr,
            "steps_per_epoch": None,
            "div_factor": self.div_factor,
        }
        self.cfg_mock = DictConfig(
            {
                "optimizer": {"_target_": "torch.optim.Adam", "lr": self.lr, "betas": self.betas.copy()},
                "lightning": {
                    "trainer": {
                        "overfitting": {
                            "enable": False,
                            "params": {
                                "overfit_batches": 1,
                            },
                        },
                        "params": {
                            "max_epochs": self.max_epochs,
                        },
                    },
                    "distributed_training": {"equal_variance_scaling_strategy": False},
                },
                "dataloader": {"params": {"batch_size": self.batch_size}},
                "warm_up_scheduler": {"lr_lambda": {"warm_up_steps": 0.0}},
            }
        )

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_optimizer_config_equal_variance(self) -> None:
        """Test default setting where the lr is scaled to maintain equal variance."""
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lightning.distributed_training.equal_variance_scaling_strategy = True

        # test method of scaling lr to maintain equal_variance
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        msg = f"Expected {(self.world_size**0.5)*self.lr} but got {cfg_mock.optimizer.lr}"
        msg_beta_1 = f"Expected {self.betas[0]}, {self.world_size ** 0.5}, {self.betas[0]**(self.world_size ** 0.5)} but got {cfg_mock.optimizer.betas[0]}"
        msg_beta_2 = f"Expected {self.betas[1]**(self.world_size ** 0.5)} but got {cfg_mock.optimizer.betas[1]}"

        self.assertAlmostEqual(
            float(cfg_mock.optimizer.lr),
            (self.world_size**0.5) * self.lr,
            msg=msg,
        )
        self.assertAlmostEqual(
            float(cfg_mock.optimizer.betas[0]),
            self.betas[0] ** (self.world_size**0.5),
            msg=msg_beta_1,
        )
        self.assertAlmostEqual(
            float(cfg_mock.optimizer.betas[1]),
            self.betas[1] ** (self.world_size**0.5),
            msg=msg_beta_2,
        )

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_optimizer_config_linearly(self) -> None:
        """Test default setting where the lr is scaled linearly."""
        cfg_mock = self.cfg_mock.copy()

        # test method of scaling lr linearly
        cfg_mock = update_distributed_optimizer_config(cfg_mock)

        msg = f"Expected {self.world_size*self.lr} but got {cfg_mock.optimizer.lr}"
        msg_beta_1 = f"Expected {self.betas[0]**self.world_size} but got {cfg_mock.optimizer.betas[0]}"
        msg_beta_2 = f"Expected {self.betas[1]**self.world_size} but got {cfg_mock.optimizer.betas[1]}"

        self.assertAlmostEqual(float(cfg_mock.optimizer.lr), self.world_size * self.lr, msg=msg)
        self.assertAlmostEqual(
            float(cfg_mock.optimizer.betas[0]),
            (self.betas[0] ** (self.world_size)),
            msg=msg_beta_1,
        )
        self.assertAlmostEqual(
            float(cfg_mock.optimizer.betas[1]),
            (self.betas[1] ** (self.world_size)),
            msg=msg_beta_2,
        )

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_not_one_cycle_lr(self) -> None:
        """
        Test default setting where the lr_scheduler is not supported.
        Currently, anything other than OneCycleLR is not supported.
        """
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lr_scheduler = self.exponential_lr_scheduler_cfg.copy()
        cfg_mock.lightning.trainer.overfitting.enable = True
        cfg_mock.lightning.trainer.overfitting.params.overfit_batches = 1

        # test that the steps_per_epoch attribute of the cfg was not edited
        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)
        msg_steps_per_epoch = f"Expected Mock to not be edited, but steps_per_epoch was edited: steps_per_epoch is {cfg_mock.lr_scheduler.steps_per_epoch}"

        self.assertIsNone(cfg_mock.lr_scheduler.steps_per_epoch, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_oclr_overfit_zero_batches(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 0."""
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lr_scheduler = self.one_cycle_lr_scheduler_cfg.copy()
        cfg_mock.lightning.trainer.overfitting.enable = True
        cfg_mock.lightning.trainer.overfitting.params.overfit_batches = 0

        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)

        expected_steps_per_epoch = math.ceil(math.ceil(self.num_train_batches / self.world_size) / self.max_epochs)

        msg_steps_per_epoch = (
            f"Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}"
        )
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_overfit_one_batches(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 1."""
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lr_scheduler = self.one_cycle_lr_scheduler_cfg.copy()
        cfg_mock.lightning.trainer.overfitting.enable = True
        cfg_mock.lightning.trainer.overfitting.params.overfit_batches = 1

        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)

        expected_steps_per_epoch = math.ceil(
            cfg_mock.lightning.trainer.overfitting.params.overfit_batches / self.world_size / self.max_epochs
        )

        msg_steps_per_epoch = (
            f"Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}"
        )

        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)

    @patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}, clear=True)
    def test_update_distributed_lr_scheduler_config_overfit_batches_fractional(self) -> None:
        """Test default setting where the overfit_batches parameter is set to 1."""
        cfg_mock = self.cfg_mock.copy()
        cfg_mock.lr_scheduler = self.one_cycle_lr_scheduler_cfg.copy()
        cfg_mock.lightning.trainer.overfitting.enable = True
        cfg_mock.lightning.trainer.overfitting.params.overfit_batches = 0.5

        cfg_mock = update_distributed_lr_scheduler_config(cfg_mock, num_train_batches=self.num_train_batches)

        batches_to_overfit = math.ceil(
            self.num_train_batches * cfg_mock.lightning.trainer.overfitting.params.overfit_batches
        )
        expected_steps_per_epoch = math.ceil(math.ceil(batches_to_overfit / self.world_size) / self.max_epochs)

        msg_steps_per_epoch = (
            f"Expected steps per epoch to be {expected_steps_per_epoch} but got {cfg_mock.lr_scheduler.steps_per_epoch}"
        )
        self.assertEqual(cfg_mock.lr_scheduler.steps_per_epoch, expected_steps_per_epoch, msg=msg_steps_per_epoch)


if __name__ == "__main__":
    unittest.main()
