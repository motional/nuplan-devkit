import abc


class AbstractMainCallback(abc.ABC):
    """Abstract class for main function callbacks."""

    def on_run_simulation_start(self) -> None:
        """Callback after the simulation function starts."""
        pass

    def on_run_simulation_end(self) -> None:
        """Callback before the simulation function ends."""
        pass
