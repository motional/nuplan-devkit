import traceback
from functools import partial
from typing import Any, Iterable, Iterator, List, Tuple, cast

import ray
from nuplan.planning.utils.multithreading.worker_pool import Task
from ray.exceptions import RayTaskError
from ray.remote_function import RemoteFunction
from tqdm import tqdm


def _ray_object_iterator(initial_ids: List[ray.ObjectRef]) -> Iterator[Tuple[ray.ObjectRef, Any]]:
    """
    Iterator that waits for each ray object in the input object list to be completed and fetches the result.
    :param initial_ids: list of ray object ids
    :yield: result of worker
    """
    next_ids = initial_ids

    while next_ids:
        ready_ids, not_ready_ids = ray.wait(next_ids)
        next_id = ready_ids[0]

        yield next_id, ray.get(next_id)

        next_ids = not_ready_ids


def _ray_map_items(task: Task, *item_lists: Iterable[List[Any]]) -> List[Any]:
    """
    Map each item of a list of arguments to a callable and executes in parallel.
    :param fn: callable to be run
    :param item_list: items to be parallelized
    :return: list of outputs
    """
    assert len(item_lists) > 0, 'No map arguments received for mapping'
    assert all(isinstance(items, list) for items in item_lists), 'All map arguments must be lists'
    assert all(len(cast(List, items)) == len(item_lists[0])  # type: ignore
               for items in item_lists), 'All lists must have equal size'
    fn = task.fn
    # Wrap function in remote decorator and create ray objects
    if isinstance(fn, partial):
        _, _, pack = fn.__reduce__()  # type: ignore
        fn, _, args, _ = pack
        remote_fn: RemoteFunction = ray.remote(fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids = [remote_fn.remote(*items, **args) for items in zip(*item_lists)]
    else:
        remote_fn: RemoteFunction = ray.remote(fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids = [remote_fn.remote(*items) for items in zip(*item_lists)]

    # Create ordered map to track order of objects inserted in the queue
    object_result_map = dict.fromkeys(object_ids, None)

    # Asynchronously iterate through the object and track progress
    for object_id, output in tqdm(_ray_object_iterator(object_ids), total=len(object_ids), desc='Ray objects'):
        object_result_map[object_id] = output

    results = list(object_result_map.values())

    return results


def ray_map(task: Task, *item_lists: Iterable[List[Any]]) -> List[Any]:
    """
    Initialize ray, align item lists and map each item of a list of arguments to a callable and executes in parallel.
    :param task: callable to be run
    :param item_lists: items to be parallelized
    :return: list of outputs
    """

    try:
        results = _ray_map_items(task, *item_lists)
        return results
    except (RayTaskError, Exception) as exc:
        ray.shutdown()
        traceback.print_exc()
        raise RuntimeError(exc)
