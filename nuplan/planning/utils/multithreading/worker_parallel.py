import concurrent
import concurrent.futures
import logging
from concurrent.futures import Future
from typing import Any, Iterable, List, Optional

from tqdm import tqdm

from nuplan.planning.utils.multithreading.worker_pool import (
    Task,
    WorkerPool,
    WorkerResources,
    get_max_size_of_arguments,
)

logger = logging.getLogger(__name__)


class SingleMachineParallelExecutor(WorkerPool):
    """
    This worker distributes all tasks across multiple threads on this machine.
    """

    def __init__(self, use_process_pool: bool = False, max_workers: Optional[int] = None):
        """
        Create worker with limited threads.
        :param use_process_pool: if true, ProcessPoolExecutor will be used as executor, otherwise ThreadPoolExecutor.
        :param max_workers: if available, use this number as used number of threads.
        """
        # Set the number of available threads
        number_of_cpus_per_node = max_workers if max_workers else WorkerResources.current_node_cpu_count()

        super().__init__(
            WorkerResources(
                number_of_nodes=1, number_of_cpus_per_node=number_of_cpus_per_node, number_of_gpus_per_node=0
            )
        )

        # Create executor
        self._executor = (
            concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            if use_process_pool
            else concurrent.futures.ThreadPoolExecutor(max_workers=number_of_cpus_per_node)
        )

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """Inherited, see superclass."""
        return list(
            tqdm(
                self._executor.map(task.fn, *item_lists),
                leave=False,
                total=get_max_size_of_arguments(*item_lists),
                desc='SingleMachineParallelExecutor',
                disable=not verbose,
            )
        )

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        return self._executor.submit(task.fn, *args, **kwargs)
