from typing import Any, Callable, List, Optional

import numpy as np
from psutil import cpu_count

from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool


def chunk_list(input_list: List[Any], num_chunks: Optional[int] = None) -> List[List[Any]]:
    """
    Chunks a list to equal sized lists. The size of the last list might be truncated.
    :param input_list: List to be chunked.
    :param num_chunks: Number of chunks, equals to the number of cores if set to None.
    :return: List of equal sized lists.
    """
    num_chunks = num_chunks if num_chunks else cpu_count(logical=True)
    chunks = np.array_split(input_list, num_chunks)  # type: ignore
    return [chunk.tolist() for chunk in chunks if len(chunk) != 0]


def worker_map(worker: WorkerPool, fn: Callable[..., List[Any]], input_objects: List[Any]) -> List[Any]:
    """
    Map a list of objects through a worker.
    :param worker: Worker pool to use for parallelization.
    :param fn: Function to use when mapping.
    :param input_objects: List of objects to map.
    :return: List of mapped objects.
    """
    if worker.number_of_threads == 0:
        return fn(input_objects)

    object_chunks = chunk_list(input_objects, worker.number_of_threads)
    scattered_objects = worker.map(Task(fn=fn), object_chunks)
    output_objects = [result for results in scattered_objects for result in results]

    return output_objects
