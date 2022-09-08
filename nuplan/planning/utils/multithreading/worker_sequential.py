import logging
from concurrent.futures import Future
from typing import Any, Iterable, List

from tqdm import tqdm

from nuplan.planning.utils.multithreading.worker_pool import (
    Task,
    WorkerPool,
    WorkerResources,
    get_max_size_of_arguments,
)

logger = logging.getLogger(__name__)


class Sequential(WorkerPool):
    """
    This function does execute all functions sequentially.
    """

    def __init__(self) -> None:
        """
        Initialize simple sequential worker.
        """
        super().__init__(WorkerResources(number_of_nodes=1, number_of_cpus_per_node=1, number_of_gpus_per_node=0))

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """Inherited, see superclass."""
        if task.num_cpus not in [None, 1]:
            raise ValueError(f'Expected num_cpus to be 1 or unset for Sequential worker, got {task.num_cpus}')
        output = [
            task.fn(*args)
            for args in tqdm(
                zip(*item_lists),
                leave=False,
                total=get_max_size_of_arguments(*item_lists),
                desc='Sequential',
                disable=not verbose,
            )
        ]
        return output

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        raise NotImplementedError
