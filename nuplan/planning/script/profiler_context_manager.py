import pathlib
from types import TracebackType

from nuplan.planning.training.callbacks.profile_callback import ProfileCallback


class ProfilerContextManager:
    """
    Class to wrap calls with a profiler callback.
    """

    def __init__(self, output_dir: str, enable_profiling: bool, name: str):
        """
        Build a profiler context.
        :param output_dir: dir to save profiling results in
        :param enable_profiling: whether we have profiling enabled or not
        :param name: name of the code segment we are profiling
        """
        self.profiler = ProfileCallback(pathlib.Path(output_dir)) if enable_profiling else None
        self.name = name

    def __enter__(self) -> None:
        """Start the profiler context."""
        if self.profiler:
            self.profiler.start_profiler(self.name)

    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        """
        Stop the profiler context and save the results.
        :param exc_type: type of exception raised while context is active
        :param exc_val: value of exception raised while context is active
        :param exc_tb: traceback of exception raised while context is active
        """
        if self.profiler:
            self.profiler.save_profiler(self.name)
