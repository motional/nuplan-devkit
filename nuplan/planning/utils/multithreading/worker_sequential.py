import logging
from typing import Any, Iterable, List

from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool, WorkerResources, \
    get_max_size_of_arguments
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Sequential(WorkerPool):
    """
    This function does execute all functions sequentially
    """

    def __init__(self):
        """
        Initialize simple sequential worker
        """
        super().__init__(WorkerResources(number_of_nodes=1, number_of_cpus_per_node=1, number_of_gpus_per_node=0))

    def _map(self, task: Task, *item_lists: Iterable[List[Any]]) -> List[Any]:
        """ Inherited, see superclass. """
        output = [task.fn(*args) for args in
                  tqdm(zip(*item_lists), leave=False, total=get_max_size_of_arguments(*item_lists), desc="Sequential")]
        return output
