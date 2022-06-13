import logging
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, cast
from uuid import uuid1

import ray
from ray.exceptions import RayTaskError
from ray.remote_function import RemoteFunction
from tqdm import tqdm

from nuplan.planning.utils.multithreading.worker_pool import Task


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


def wrap_function(fn: Callable[..., Any], log_dir: Optional[Path] = None) -> Callable[..., Any]:
    """
    Wraps a function to save its logs to a unique file inside the log directory.
    :param fn: function to be wrapped.
    :param log_dir: directory to store logs (wrapper function does nothing if it's not set).
    :return: wrapped function which changes logging settings while it runs.
    """

    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if log_dir is None:
            return fn(*args, **kwargs)

        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{uuid1().hex}__{fn.__name__}.log'

        logging.basicConfig()
        logger = logging.getLogger()
        fh = logging.FileHandler(log_path, delay=True)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

        # Silent botocore which is polluting the terminal because of serialization and deserialization
        # with following message: INFO:botocore.credentials:Credentials found in config file: ~/.aws/config
        logging.getLogger('botocore').setLevel(logging.WARNING)

        result = fn(*args, **kwargs)

        fh.flush()
        fh.close()
        logger.removeHandler(fh)

        return result

    return wrapped_fn


def _ray_map_items(task: Task, *item_lists: Iterable[List[Any]], log_dir: Optional[Path] = None) -> List[Any]:
    """
    Map each item of a list of arguments to a callable and executes in parallel.
    :param fn: callable to be run
    :param item_list: items to be parallelized
    :param log_dir: directory to store worker logs
    :return: list of outputs
    """
    assert len(item_lists) > 0, 'No map arguments received for mapping'
    assert all(isinstance(items, list) for items in item_lists), 'All map arguments must be lists'
    assert all(
        len(cast(List, items)) == len(item_lists[0]) for items in item_lists  # type: ignore
    ), 'All lists must have equal size'
    fn = task.fn
    # Wrap function in remote decorator and create ray objects
    if isinstance(fn, partial):
        _, _, pack = fn.__reduce__()  # type: ignore
        fn, _, args, _ = pack
        fn = wrap_function(fn, log_dir=log_dir)
        remote_fn: RemoteFunction = ray.remote(fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids = [remote_fn.remote(*items, **args) for items in zip(*item_lists)]
    else:
        fn = wrap_function(fn, log_dir=log_dir)
        remote_fn = ray.remote(fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids = [remote_fn.remote(*items) for items in zip(*item_lists)]

    # Create ordered map to track order of objects inserted in the queue
    object_result_map = dict.fromkeys(object_ids, None)

    # Asynchronously iterate through the object and track progress
    for object_id, output in tqdm(_ray_object_iterator(object_ids), total=len(object_ids), desc='Ray objects'):
        object_result_map[object_id] = output

    results = list(object_result_map.values())

    return results


def ray_map(task: Task, *item_lists: Iterable[List[Any]], log_dir: Optional[Path] = None) -> List[Any]:
    """
    Initialize ray, align item lists and map each item of a list of arguments to a callable and executes in parallel.
    :param task: callable to be run
    :param item_lists: items to be parallelized
    :param log_dir: directory to store worker logs
    :return: list of outputs
    """
    try:
        results = _ray_map_items(task, *item_lists, log_dir=log_dir)
        return results
    except (RayTaskError, Exception) as exc:
        ray.shutdown()
        traceback.print_exc()
        raise RuntimeError(exc)
