from dataclasses import dataclass
from typing import Any, List


@dataclass
class SplitState:
    """Dataclass representing a state split between fixed states, linear states and angular states."""

    linear_states: List[Any]  # Variable states
    angular_states: List[float]  # Variable states, representing angles, with 2pi period
    fixed_states: List[Any]  # Constant states

    def __len__(self) -> int:
        """Returns the number of states"""
        return len(self.linear_states) + len(self.angular_states) + len(self.fixed_states)
