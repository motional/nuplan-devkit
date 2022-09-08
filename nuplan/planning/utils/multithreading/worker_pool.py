import abc
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

from psutil import cpu_count

logger = logging.getLogger(__name__)


def get_max_size_of_arguments(*item_lists: Iterable[List[Any]]) -> int:
    """
    Find the argument with most elements.
        e.g. [db, [arg1, arg2] -> 2.
    :param item_lists: arguments where some of the arguments is a list.
    :return: size of largest list.
    """
    lengths = [len(items) for items in item_lists if isinstance(items, list)]
    if len(list(set(lengths))) > 1:
        raise RuntimeError(f'There exists lists with different element size = {lengths}!')
    return max(lengths) if len(lengths) != 0 else 1


def align_size_of_arguments(*item_lists: Iterable[List[Any]]) -> Tuple[int, Iterable[List[Any]]]:
    """
    Align item lists by repeating elements in order to achieve the same size.
        eg. [db, [arg1, arg2] -> [[db, db], [arg1, arg2]].
    :param item_lists: multiple arguments which will be used to call a function.
    :return: arguments with same dimension, e.g., [[db, db], [arg1, arg1]].
    """
    max_size = get_max_size_of_arguments(*item_lists)
    aligned_item_lists = [items if isinstance(items, list) else [items] * max_size for items in item_lists]
    return max_size, aligned_item_lists


@dataclass(frozen=True)
class Task:
    """This class represents a task that can be submitted to a worker with specific resource requirements."""

    fn: Callable[..., Any]  # Function that should be called with arguments

    # Number of CPUs required for this task
    # if None, the default number of CPUs will be used that were specified in initialization
    num_cpus: Optional[int] = None

    # Fraction of GPUs required for this task, this number can be also smaller than 1
    # It num_gpus is smaller than 1, only a part of a GPU is allocated to this task.
    # NOTE: it is the responsibility of a user to make sure that a model fits into num_gpus
    # if None, no GPU will be used
    num_gpus: Optional[Union[int, float]] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call function with args.
        :return: output from fn.
        """
        return self.fn(*args, **kwargs)


@dataclass(frozen=True)
class WorkerResources:
    """Data class to indicate resources used by workers."""

    number_of_nodes: int  # Number of available independent nodes
    number_of_gpus_per_node: int  # Number of GPUs per node
    # Number of CPU logical cores per node, this can be smaller than
    # current_node_cpu_count if user will decide to limit it
    number_of_cpus_per_node: int

    @property
    def number_of_threads(self) -> int:
        """
        :return: the number of available threads across all nodes.
        """
        return self.number_of_nodes * self.number_of_cpus_per_node

    @staticmethod
    def current_node_cpu_count() -> int:
        """
        :return: the number of logical cores on the current machine.
        """
        return cpu_count(logical=True)  # type: ignore


class WorkerPool(abc.ABC):
    """
    This class executed function on list of arguments. This can either be distributed/parallel or sequential worker.
    """

    def __init__(self, config: WorkerResources):
        """
        Initialize worker with resource description.
        :param config: setup of this worker.
        """
        self.config = config

        if self.config.number_of_threads < 1:
            raise RuntimeError(f'Number of threads can not be 0, and it is {self.config.number_of_threads}!')

        logger.info(f'Worker: {self.__class__.__name__}')
        logger.info(f'{self}')

    def map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """
        Run function with arguments from item_lists, this function will make sure all arguments have the same
        number of elements.
        :param task: function to be run.
        :param item_lists: arguments to the function.
        :param verbose: Whether to increase logger verbosity.
        :return: type from the fn.
        """
        max_size, aligned_item_lists = align_size_of_arguments(*item_lists)

        if verbose:
            logger.info(f'Submitting {max_size} tasks!')
        return self._map(task, *aligned_item_lists, verbose=verbose)

    @abc.abstractmethod
    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """
        Run function with arguments from item_lists. This function can assume that all the args in item_lists have
        the same number of elements.
        :param fn: function to be run.
        :param item_lists: arguments to the function.
        :param number_of_elements: number of calls to the function.
        :return: type from the fn.
        """

    @abc.abstractmethod
    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """
        Submit a task to the worker.
        :param task: to be submitted.
        :param args: arguments for the task.
        :param kwargs: keyword arguments for the task.
        :return: future.
        """
        pass

    @property
    def number_of_threads(self) -> int:
        """
        :return: the number of available threads across all nodes.
        """
        return self.config.number_of_threads

    def __str__(self) -> str:
        """
        :return: string with information about this worker.
        """
        return (
            f'Number of nodes: {self.config.number_of_nodes}\n'
            f'Number of CPUs per node: {self.config.number_of_cpus_per_node}\n'
            f'Number of GPUs per node: {self.config.number_of_gpus_per_node}\n'
            f'Number of threads across all nodes: {self.config.number_of_threads}'
        )
