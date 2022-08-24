import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


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


def suppress_geopandas_warning() -> None:
    """
    Filters warning message about incompatible PyGEOS verions. The underlying cause is a pip packaging error that will
    hopefully be fixed in the future, and available workarounds complicate installation. Must run before geopandas import.
    It's possible for the error to still appear if geopandas is imported elsewhere, but this covers the main case when
    running simulations.
    """
    warnings.filterwarnings('ignore', '.*The Shapely GEOS version .* is incompatible with the GEOS version PyGEOS.*')
