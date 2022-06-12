from dataclasses import dataclass
from typing import List


@dataclass
class IDMAgentState:
    """IDM smart-agent state."""

    progress: float  # [m] distane a long a path
    velocity: float  # [m/s] velocity along the oath

    def to_array(self) -> List[float]:
        """Return agent state as an array."""
        return [self.progress, self.velocity]


@dataclass
class IDMLeadAgentState(IDMAgentState):
    """IDM smart-agent state."""

    length_rear: float  # [m] length from vehicle CoG to the rear bumper

    def to_array(self) -> List[float]:
        """Return agent state as an array."""
        return [self.progress, self.velocity, self.length_rear]
