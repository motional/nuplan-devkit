import unittest
from time import sleep
from typing import Any, Tuple

import torch.cuda

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.raster_model import RasterModel
from nuplan.planning.utils.multithreading.worker_pool import Task
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed


def function_to_load_model(dummy_var: Any) -> Tuple[bool, int]:
    """
    Dummy function
    return: gpu_available, num_threads avaialble for torch
    """
    model = RasterModel(
        feature_builders=[],
        target_builders=[],
        model_name='resnet50',
        pretrained=True,
        num_input_channels=4,
        num_features_per_pose=3,
        future_trajectory_sampling=TrajectorySampling(num_poses=10, time_horizon=5),
    )
    gpu_available = torch.cuda.is_available()
    device = torch.device('cuda' if gpu_available else 'cpu')

    model.to(device)
    sleep(1)

    return gpu_available, torch.get_num_threads()


class TestWorkerPool(unittest.TestCase):
    """Unittest class for WorkerPool"""

    def setUp(self) -> None:
        """
        Setup worker
        """
        self.worker = RayDistributed(debug_mode=True)

    def test_ray(self) -> None:
        """
        Test ray GPU allocation
        """
        num_calls = 3
        num_gpus = 1
        output = self.worker.map(Task(fn=function_to_load_model, num_gpus=num_gpus), num_calls * [1])
        for gpu_available, num_threads in output:
            self.assertTrue(gpu_available)
            self.assertGreater(num_threads, 0)


if __name__ == '__main__':
    unittest.main()
