import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class TimingCallback(AbstractCallback):
    """Callback to log timing information to Tensorboard as the simulation runs."""

    def __init__(self, writer: SummaryWriter):
        """
        Constructor for TimingCallback.
        :param writer: handler for writing to tensorboard.
        """
        # Tensorboard
        self._writer = writer

        # Capture all the timings
        self._scenarios_captured: Dict[str, Any] = defaultdict(None)

        # Memory for starting time in order to compute elapsed time
        self._step_start: Optional[float] = None
        self._simulation_start: Optional[float] = None
        self._planner_start: Optional[float] = None

        # Accumulation of all the steps
        self._step_duration: List[float] = []
        self._planner_step_duration: List[float] = []

        # Current step
        self._tensorboard_global_step = 0

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        self._planner_start = self._get_time()

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        assert self._planner_start, "Start time has to be set: on_planner_end!"
        self._planner_step_duration.append(self._get_time() - self._planner_start)

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        self._scenarios_captured[setup.scenario.token] = None
        self._simulation_start = self._get_time()

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Inherited, see superclass."""
        assert self._simulation_start, "Start time has to be set: on_simulation_end!"
        elapsed_time = self._get_time() - self._simulation_start

        timings = {
            "simulation_elapsed_time": elapsed_time,
            "mean_step_time": np.mean(self._step_duration),
            "max_step_time": np.max(self._step_duration),
            "max_planner_step_time": np.max(self._planner_step_duration),
            "mean_planner_step_time": np.mean(self._planner_step_duration),
        }

        # Publish timings
        step = self._tensorboard_global_step
        self._writer.add_scalar("simulation_elapsed_time", timings["simulation_elapsed_time"], step)
        self._writer.add_scalar("mean_step_time", timings["mean_step_time"], step)
        self._writer.add_scalar("max_step_time", timings["max_step_time"], step)
        self._writer.add_scalar("max_planner_step_time", timings["max_planner_step_time"], step)
        self._writer.add_scalar("mean_planner_step_time", timings["mean_planner_step_time"], step)
        self._tensorboard_global_step += 1

        # Store timings
        self._scenarios_captured[setup.scenario.token] = timings

        # Erase history
        self._step_duration = []
        self._planner_step_duration = []

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        self._step_start = self._get_time()

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        assert self._step_start, "Start time has to be set: on_step_end!"
        elapsed_time = self._get_time() - self._step_start
        self._step_duration.append(elapsed_time)

    def _get_time(self) -> float:
        return time.perf_counter()
