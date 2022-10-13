import unittest

import torch
from omegaconf import DictConfig

from nuplan.planning.script.builders.lr_scheduler_builder import build_lr_scheduler


class TestLRSchedulerBuilder(unittest.TestCase):
    """Test update_distributed_optimizer_config function."""

    world_size = 4

    def setUp(self) -> None:
        """Setup test attributes."""
        self.lr = 5e-5
        self.weight_decay = 5e-4
        self.betas = [0.9, 0.999]
        self.mock_params = [torch.rand(1)]

        self.warm_up_lr_scheduler_cfg = DictConfig(
            {
                '_target_': 'torch.optim.lr_scheduler.LambdaLR',
                '_convert_': 'all',
                'optimizer': '',
                'lr_lambda': {
                    '_target_': 'nuplan.planning.script.builders.lr_scheduler_builder.get_warm_up_lr_scheduler_func',
                    'warm_up_steps': 100,
                    'warm_up_strategy': 'linear',
                },
            }
        )
        self.lr_scheduler_cfg = DictConfig(
            {
                '_target_': 'torch.optim.lr_scheduler.OneCycleLR',
                '_convert_': 'all',
                'optimizer': '',
                'max_lr': 5e-5,
                'epochs': 1,
                'steps_per_epoch': 100,
                'pct_start': 0.25,
                'anneal_strategy': 'cos',
                'cycle_momentum': True,
                'base_momentum': 0.85,
                'max_momentum': 0.95,
                'div_factor': 10,
                'final_div_factor': 10,
                'last_epoch': -1,
            }
        )
        self.initial_lr = self.lr / self.lr_scheduler_cfg.div_factor

    def _get_lr_from_optimizer(self, optimizer: torch.optim.Optimizer) -> float:
        lr: float
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def test_build_lr_scheduler_with_warm_up_scheduler_and_one_cycle_lr_scheduler(self) -> None:
        """Test that lr_scheduler with warm up scheduler works as expected"""
        optimizer = torch.optim.AdamW(
            lr=self.lr, weight_decay=self.weight_decay, betas=self.betas, params=self.mock_params
        )

        sequential_scheduler = build_lr_scheduler(
            optimizer=optimizer,
            lr=self.lr,
            warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler_cfg,
            lr_scheduler_cfg=self.lr_scheduler_cfg,
        )['scheduler']

        # Check that optimizer initial lr is 0
        self.assertAlmostEqual(self._get_lr_from_optimizer(sequential_scheduler.optimizer), 0.0)

        total_steps = (
            self.lr_scheduler_cfg.steps_per_epoch * self.lr_scheduler_cfg.epochs
            + self.warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps
        )
        lrs = []
        for _ in range(total_steps):
            sequential_scheduler.step()
            lrs.append(self._get_lr_from_optimizer(sequential_scheduler.optimizer))

        lr_at_end_of_warm_up = lrs[self.warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps - 1]
        lr_at_end_of_training = lrs[-1]
        max_lr = max(lrs)

        self.assertAlmostEqual(max_lr, self.lr)
        self.assertAlmostEqual(lr_at_end_of_warm_up, self.initial_lr)
        self.assertAlmostEqual(lr_at_end_of_training, self.initial_lr / self.lr_scheduler_cfg.final_div_factor)

    def test_build_lr_scheduler_with_warm_up_scheduler_and_no_main_scheduler(self) -> None:
        """Test that lr_scheduler with warm up scheduler works as expected"""
        optimizer = torch.optim.AdamW(
            lr=self.lr, weight_decay=self.weight_decay, betas=self.betas, params=self.mock_params
        )

        sequential_scheduler = build_lr_scheduler(
            optimizer=optimizer,
            lr=self.lr,
            warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler_cfg,
            lr_scheduler_cfg=None,
        )['scheduler']

        # Check that optimizer initial lr was adjusted correctly
        self.assertAlmostEqual(self._get_lr_from_optimizer(sequential_scheduler.optimizer), 0.0)

        total_steps = (
            self.lr_scheduler_cfg.steps_per_epoch * self.lr_scheduler_cfg.epochs
            + self.warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps
        )
        lrs = []
        for _ in range(total_steps):
            sequential_scheduler.step()
            lrs.append(self._get_lr_from_optimizer(sequential_scheduler.optimizer))

        lr_at_end_of_warm_up = lrs[self.warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps - 1]
        lr_at_end_of_training = lrs[-1]

        self.assertAlmostEqual(lr_at_end_of_warm_up, self.lr)
        self.assertAlmostEqual(lr_at_end_of_training, self.lr)


if __name__ == '__main__':
    unittest.main()
