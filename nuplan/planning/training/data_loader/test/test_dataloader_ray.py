import unittest

from nuplan.planning.training.data_loader.test.skeleton_test_dataloader import SkeletonTestDataloader
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed


class TestDataloaderRay(SkeletonTestDataloader):
    """
    Tests data loading functionality in ray.
    """

    def test_dataloader_nuplan_ray(self) -> None:
        """
        Test dataloader using nuPlan DB.
        """
        self._test_dataloader(RayDistributed())


if __name__ == '__main__':
    unittest.main()
