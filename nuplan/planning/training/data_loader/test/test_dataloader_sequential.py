import unittest

from nuplan.planning.training.data_loader.test.skeleton_test_dataloader import SkeletonTestDataloader
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


class TestDataloaderSequential(SkeletonTestDataloader):
    """
    Tests data loading functionality in a sequential manner.
    """

    def test_dataloader_nuplan_sequential(self) -> None:
        """
        Test dataloader using nuPlan DB using a sequential worker.
        """
        self._test_dataloader(Sequential())


if __name__ == '__main__':
    unittest.main()
