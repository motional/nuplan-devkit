import logging
from typing import List

from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

logger = logging.getLogger(__name__)


class MultiMainCallback(AbstractMainCallback):
    """
    Combines a set of of AbstractMainCallbacks.
    """

    def __init__(self, main_callbacks: List[AbstractMainCallback]):
        """
        Callback to handle a list of main callbacks.
        :param main_callbacks: A list of main callbacks.
        """
        self._main_callbacks = main_callbacks

    def __len__(self) -> int:
        """Support len() as counting the number of callbacks."""
        return len(self._main_callbacks)

    def on_run_simulation_start(self) -> None:
        """Callback after the simulation function starts."""
        for main_callback in self._main_callbacks:
            main_callback.on_run_simulation_start()

    def on_run_simulation_end(self) -> None:
        """Callback before the simulation function ends."""
        for main_callback in self._main_callbacks:
            main_callback.on_run_simulation_end()
