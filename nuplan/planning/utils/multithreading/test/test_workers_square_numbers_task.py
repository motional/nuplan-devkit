import unittest
from typing import Any, Dict, List, Tuple, Union

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


def square_fn(x: Union[int, float]) -> Union[int, float]:
    """
    Returns the square, x^2, of a numeric input, x.
    We define it outside of the test function / test class so it can be pickled by MultiProcessing.
    """
    return x**2


class TestWorkerSquareNumbersTask(unittest.TestCase):
    """Class to run all workers on a simple worker task, squaring a list of numbers."""

    def setUp(self) -> None:
        """
        Instantiate all workers we want to check, not just numerically but for correct bazel BUILD file setup.
        """
        self.worker_arg_tuples: List[Tuple[WorkerPool, Dict[str, Any]]] = [
            (SingleMachineParallelExecutor, {}),
            (SingleMachineParallelExecutor, {"use_process_pool": True}),
            (RayDistributed, {"threads_per_node": 2}),
            (Sequential, {}),
        ]

        self.number_of_tasks = 10

    def test_square_numbers_task(self) -> None:
        """Make sure all workers can correctly execute map to square a list of numbers."""
        task_list = [x for x in range(self.number_of_tasks)]
        expected_result = [x**2 for x in task_list]

        for worker_arg_tuple in self.worker_arg_tuples:
            worker = worker_arg_tuple[0](**worker_arg_tuple[1])
            worker_result = worker.map(Task(fn=square_fn), task_list)
            self.assertEqual(worker_result, expected_result)

            if isinstance(worker, RayDistributed):
                worker.shutdown()


if __name__ == "__main__":
    unittest.main()
