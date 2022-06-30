from dataclasses import dataclass
from typing import List


@dataclass
class Color:
    """
    Represents a color.
    """

    red: float  # Red component of the color.
    green: float  # Green component of the color.
    blue: float  # Blue component of the color.
    alpha: float  # Alpha channel intensity.

    def to_list(self) -> List[float]:
        """
        Returns the color as a list of floats.
        """
        return [self.red, self.green, self.blue, self.alpha]
