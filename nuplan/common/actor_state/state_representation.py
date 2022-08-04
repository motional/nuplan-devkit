from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Union

import numpy as np
import numpy.typing as npt


@dataclass
class TimePoint:
    """
    Time instance in a time series
    """

    time_us: int  # [micro seconds] time since epoch in micro seconds

    def __post_init__(self) -> None:
        """
        Validate class after creation.
        """
        assert self.time_us >= 0, "Time point has to be positive!"

    @property
    def time_s(self) -> float:
        """
        :return [s] time in seconds.
        """
        return self.time_us * 1e-6

    def __add__(self, other: TimePoint) -> TimePoint:
        """
        Add two time points
        :param other: time point
        :return: self + other
        """
        return TimePoint(self.time_us + other.time_us)

    def __sub__(self, other: TimePoint) -> TimePoint:
        """
        Subtract two time points
        :param other: time point
        :return: self - other
        """
        return TimePoint(self.time_us - other.time_us)

    def __gt__(self, other: TimePoint) -> bool:
        """
        Self is greater than other
        :param other: time point
        :return: True if self > other, False otherwise
        """
        return self.time_us > other.time_us

    def __ge__(self, other: TimePoint) -> bool:
        """
        Self is greater or equal than other
        :param other: time point
        :return: True if self >= other, False otherwise
        """
        return self.time_us >= other.time_us

    def __lt__(self, other: TimePoint) -> bool:
        """
        Self is less than other
        :param other: time point
        :return: True if self < other, False otherwise
        """
        return self.time_us < other.time_us

    def __le__(self, other: TimePoint) -> bool:
        """
        Self is less or equal than other
        :param other: time point
        :return: True if self <= other, False otherwise
        """
        return self.time_us <= other.time_us

    def __eq__(self, other: object) -> bool:
        """
        Self is equal to other
        :param other: time point
        :return: True if self == other, False otherwise
        """
        if not isinstance(other, TimePoint):
            return NotImplemented

        return self.time_us == other.time_us

    def __hash__(self) -> int:
        """
        :return: hash for this object
        """
        return hash(self.time_us)


@dataclass
class Point2D:
    """Class to represents 2D points."""

    x: float  # [m] location
    y: float  # [m] location

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y)
        """
        return iter((self.x, self.y))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y]
        """
        return np.array([self.x, self.y], dtype=np.float64)

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y))


@dataclass
class StateSE2(Point2D):
    """
    SE2 state - representing [x, y, heading]
    """

    heading: float  # [rad] heading of a state

    @property
    def point(self) -> Point2D:
        """
        Gets a point from the StateSE2
        :return: Point with x and y from StateSE2
        """
        return Point2D(self.x, self.y)

    def as_matrix(self) -> npt.NDArray[np.float32]:
        """
        :return: 3x3 2D transformation matrix representing the SE2 state.
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), self.x],
                [np.sin(self.heading), np.cos(self.heading), self.y],
                [0.0, 0.0, 1.0],
            ]
        )

    def as_matrix_3d(self) -> npt.NDArray[np.float32]:
        """
        :return: 4x4 3D transformation matrix representing the SE2 state projected to SE3.
        """
        return np.array(
            [
                [np.cos(self.heading), -np.sin(self.heading), 0.0, self.x],
                [np.sin(self.heading), np.cos(self.heading), 0.0, self.y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def distance_to(self, state: StateSE2) -> float:
        """
        Compute the euclidean distance between two points
        :param state: state to compute distance to
        :return distance between two points
        """
        return float(np.hypot(self.x - state.x, self.y - state.y))

    @staticmethod
    def from_matrix(matrix: npt.NDArray[np.float32]) -> StateSE2:
        """
        :param matrix: 3x3 2D transformation matrix
        :return: StateSE2 object
        """
        assert matrix.shape == (3, 3), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

        vector = [matrix[0, 2], matrix[1, 2], np.arctan2(matrix[1, 0], matrix[0, 0])]
        return StateSE2.deserialize(vector)

    @staticmethod
    def deserialize(vector: List[float]) -> StateSE2:
        """
        Deserialize vector into state SE2
        :param vector: serialized list of floats
        :return: StateSE2
        """
        if len(vector) != 3:
            raise RuntimeError(f'Expected a vector of size 3, got {len(vector)}')

        return StateSE2(x=vector[0], y=vector[1], heading=vector[2])

    def serialize(self) -> List[float]:
        """
        :return: list of serialized variables [X, Y, Heading]
        """
        return [self.x, self.y, self.heading]

    def __eq__(self, other: object) -> bool:
        """
        Compare two state SE2
        :param other: object
        :return: true if the objects are equal, false otherwise
        """
        if not isinstance(other, StateSE2):
            # Return NotImplemented in case the classes are not of the same type
            return NotImplemented
        return (
            math.isclose(self.x, other.x, abs_tol=1e-3)
            and math.isclose(self.y, other.y, abs_tol=1e-3)
            and math.isclose(self.heading, other.heading, abs_tol=1e-4)
        )

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y, heading)
        """
        return iter((self.x, self.y, self.heading))

    def __hash__(self) -> int:
        """
        :return: hash for this object
        """
        return hash((self.x, self.y, self.heading))


@dataclass
class ProgressStateSE2(StateSE2):
    """
    StateSE2 parameterized by progress
    """

    progress: float  # [m] distance along a path

    @staticmethod
    def deserialize(vector: List[float]) -> ProgressStateSE2:
        """
        Deserialize vector into this class
        :param vector: containing raw float numbers containing [progress, x, ,y, heading]
        :return: ProgressStateSE2 class
        """
        if len(vector) != 4:
            raise RuntimeError(f'Expected a vector of size 4, got {len(vector)}')

        return ProgressStateSE2(progress=vector[0], x=vector[1], y=vector[2], heading=vector[3])

    def __iter__(self) -> Iterable[Union[float]]:
        """
        :return: an iterator over the tuble of (progress, x, y, heading) states
        """
        return iter((self.progress, self.x, self.y, self.heading))


@dataclass
class TemporalStateSE2(StateSE2):
    """
    Representation of a temporal state
    """

    time_point: TimePoint  # state at a time

    @property
    def time_us(self) -> int:
        """
        :return: [us] time stamp in micro seconds
        """
        return self.time_point.time_us

    @property
    def time_seconds(self) -> float:
        """
        :return: [s] time stamp in seconds
        """
        return self.time_us * 1e-6


class StateVector2D:
    """Representation of vector in 2d."""

    def __init__(self, x: float, y: float):
        """
        Create StateVector2D object
        :param x: float direction
        :param y: float direction
        """
        self._x = x  # x-axis in the vector.
        self._y = y  # y-axis in the vector.

        self.array = np.array([self.x, self.y], dtype=np.float64)

    def __repr__(self) -> str:
        """
        :return: string containing representation of this class
        """
        return f'x: {self.x}, y: {self.y}'

    def __eq__(self, other: object) -> bool:
        """
        Compare other object with this class
        :param other: object
        :return: true if other state vector is the same as self
        """
        if not isinstance(other, StateVector2D):
            return NotImplemented
        return bool(np.array_equal(self.array, other.array))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y]
        """
        return self._array

    @array.setter
    def array(self, other: npt.NDArray[np.float64]) -> None:
        """Custom setter so that the object is not corrupted."""
        self._array = other
        self._x = other[0]
        self._y = other[1]

    @property
    def x(self) -> float:
        """
        :return: x float state
        """
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        """Custom setter so that the object is not corrupted."""
        self._x = x
        self._array[0] = x

    @property
    def y(self) -> float:
        """
        :return: y float state
        """
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        """Custom setter so that the object is not corrupted."""
        self._y = y
        self._array[1] = y

    def magnitude(self) -> float:
        """
        :return: magnitude of vector
        """
        return float(np.hypot(self.x, self.y))
