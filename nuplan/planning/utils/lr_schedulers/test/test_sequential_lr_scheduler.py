import unittest
from typing import Callable

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from nuplan.planning.utils.lr_schedulers.sequential_lr_scheduler import SequentialLR


class TestSequentialLRScheduler(unittest.TestCase):
    """Test update_distributed_optimizer_config function."""

    world_size = 4

    def setUp(self) -> None:
        """Set up schedulers and optimizer"""
        self.optimizer = Adam(params=[torch.tensor([1.0])], lr=1e-4)
        self.total_steps = 300
        self.milestones = [100, 200]
        # Note: the order in which we instantiate the schedulers matters because the last scheduler to be instantiated will
        # alter the initial lr of the optimizer. Hence the first lr scheduler should be instantiated last
        self.scheduler3 = LambdaLR(self.optimizer, lambda step: 1)
        self.scheduler2 = LambdaLR(self.optimizer, self._get_mock_linear_func())
        self.scheduler1 = LambdaLR(self.optimizer, self._get_mock_linear_func())
        self.schedulers = [self.scheduler1, self.scheduler2, self.scheduler3]
        # expected lrs at step 0, step 100, step 200, step 300
        self.expected_lrs = [1e-6, 1e-4, 1e-4, 1e-4]

    def _get_mock_linear_func(self) -> Callable[..., float]:
        """Gets mock linear function"""

        def _linear_func(step: int) -> float:
            num_steps = self.milestones[1] - self.milestones[0]
            return step / num_steps if step <= num_steps else 1.0

        return _linear_func

    def _get_lr_from_optimizer(self, optimizer: torch.optim.Optimizer) -> float:
        lr: float
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def test_sequential_lr_with_multiple_schedulers(self) -> None:
        """Tests that SequentialLR Scheduler works with multiple schedulers."""
        sequential_lr_scheduler = SequentialLR(self.optimizer, self.schedulers, self.milestones)
        lr = []

        # assert learning rate has been set to 0
        self.assertAlmostEqual(self._get_lr_from_optimizer(sequential_lr_scheduler.optimizer), 0.0)
        for i in range(self.total_steps):
            sequential_lr_scheduler.step()
            print(i + 1, self._get_lr_from_optimizer(sequential_lr_scheduler.optimizer))
            lr.append(self._get_lr_from_optimizer(sequential_lr_scheduler.optimizer))

        self.assertAlmostEqual(lr[0], self.expected_lrs[0])
        self.assertAlmostEqual(lr[self.milestones[0] - 1], self.expected_lrs[1])
        self.assertAlmostEqual(lr[self.milestones[1] - 1], self.expected_lrs[2])
        self.assertAlmostEqual(lr[-1], self.expected_lrs[-1])


if __name__ == '__main__':
    unittest.main()
