import functools
import hashlib
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

GenericCallable = Callable[[Any], Any]


def try_n_times(
    fn: Callable[..., Any],
    args: List[Any],
    kwargs: Dict[Any, Any],
    errors: Tuple[Any],
    max_tries: int,
    sleep_time: float = 0,
) -> Any:
    """
    Keeps calling a function with given parameters until maximum number of tries, catching a set of given errors.
    :param fn: The function to call
    :param args: Argument list
    :param kwargs" Keyword arguments
    :param errors: Expected errors to be ignored
    :param max_tries: Maximal number of tries before raising error
    :param sleep_time: Time waited between subsequent tried to the function call.
    :return: The return value of the given function
    """
    assert max_tries > 0, "Number of tries must be a positive integer"
    attempts = 0
    error = None

    while attempts < max_tries:
        try:
            return fn(*args, **kwargs)
        except errors as e:
            error = e
            attempts += 1
            logging.warning(f"Tried to call {fn} raised {e}, trying {max_tries - attempts} more times.")
            time.sleep(sleep_time)
            pass

    if error:
        raise error


def keep_trying(
    fn: Callable[..., Any],
    args: List[Any],
    kwargs: Dict[Any, Any],
    errors: Tuple[Any],
    timeout: float,
    sleep_time: float = 0.1,
) -> Any:
    """
    Keeps calling a function with given parameters until timeout (at least once), catching a set of given errors.
    :param fn: The function to call
    :param args: Argument list
    :param kwargs" Keyword arguments
    :param errors: Expected errors to be ignored
    :param timeout: Maximal time before timeout (seconds)
    :param sleep_time: Time waited between subsequent tried to the function call.
    :return: The return value of the given function
    """
    assert timeout > 0, "Timeout must be a positive real number"
    start_time = time.time()
    max_time = start_time + timeout
    first_run = True

    while time.time() < max_time or first_run:
        try:
            return fn(*args, **kwargs), time.time() - start_time
        except errors:
            first_run = False
            time.sleep(sleep_time)

    raise TimeoutError(f"Timeout on function call {fn}({args}{kwargs}) catching {errors}")


@functools.cache
def get_unique_job_id() -> str:
    """
    In the cluster, it generates a hash from the unique job ID called NUPLAN_JOB_ID.
    Locally, it generates a hash from a UUID.

    Note that the returned value is cached as soon as the function is called the first time.
    After that, it is going to return always the same value.
    If a new value is needed, use get_unique_job_id.cache_clear() first.
    """
    global_job_id_str = os.environ.get("NUPLAN_JOB_ID", str(uuid.uuid4())).encode("utf-8")
    return hashlib.sha256(global_job_id_str).hexdigest()


def static_vars(**kwargs: Any) -> GenericCallable:
    """
    Decorator to assign static variables to functions
    """

    def decorate(func: GenericCallable) -> GenericCallable:
        for key, value in kwargs.items():
            setattr(func, key, value)
        return func

    return decorate


@functools.lru_cache(maxsize=None)
@static_vars(id=-1)
def get_unique_incremental_track_id(_: str) -> int:
    """
    Generate a unique ID (increasing number)
    :return int Unique ID
    """
    get_unique_incremental_track_id.id += 1  # type: ignore
    return get_unique_incremental_track_id.id  # type: ignore
