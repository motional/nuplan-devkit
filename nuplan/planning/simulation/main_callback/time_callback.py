import logging
import time

from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

logger = logging.getLogger(__name__)


class TimeCallback(AbstractMainCallback):
    """
    Callback for tracking how long a simulation took to run.
    """

    def __init__(self) -> None:
        """Callback to log simulation duration at the end of process."""
        self._start_time = 0.0

    def on_run_simulation_start(self) -> None:
        """Callback after the simulation function starts."""
        self._start_time = time.perf_counter()

    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        end_time = time.perf_counter()
        elapsed_time_s = end_time - self._start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))
        logger.info(f"Simulation duration: {time_str} [HH:MM:SS]")
