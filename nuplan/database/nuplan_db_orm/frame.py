from enum import IntEnum


class Frame(IntEnum):
    """Coordinate frames."""

    GLOBAL = 0  # Global frame
    VEHICLE = 1  # Vehicle frame
    SENSOR = 2  # Sensor frame
