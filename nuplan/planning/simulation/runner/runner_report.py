from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from nuplan.planning.simulation.planner.abstract_planner import PlannerReport

logger = logging.getLogger(__name__)


@dataclass
class RunnerReport:
    """Report for a runner."""

    succeeded: bool  # True if simulation was successful
    error_message: Optional[str]  # None if simulation succeeded, traceback if it failed
    start_time: float  # Time simulation.run() was called
    end_time: Optional[float]  # Time simulation.run() returned, when the error was logged, or None temporarily
    planner_report: Optional[PlannerReport]  # Planner report containing stats about planner runtime, None if the
    # runner didn't run a planner (eg. MetricRunner), or when a run fails

    # Metadata about the simulations
    scenario_name: str
    planner_name: str
    log_name: str
