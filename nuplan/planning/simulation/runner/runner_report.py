from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RunnerReport:
    """Report for a runner."""

    succeeded: bool  # True if simulation was successful
    error_message: Optional[str]  # None if simulation succeeded, traceback if it failed
    start_time: float  # Time simulation.run() was called
    end_time: Optional[float]  # Time simulation.run() returned, when the error was logged, or None temporarily

    # Metadata about the simulation
    scenario_name: str
    planner_name: str
    log_name: str
