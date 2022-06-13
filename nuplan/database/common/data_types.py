"""
Common data types in template.py.
Here we have a thin wrapper on top of the original list/dict type, it is
easy to use and compatible.
"""
from enum import Enum
from typing import Any, Dict, List, Tuple


class RLE(Dict[Any, Any]):
    """RLE Properties."""

    @property
    def size(self) -> Tuple[int, int]:
        """
        Get size.
        :return: Size.
        """
        return self['size']  # type: ignore

    @property
    def counts(self) -> bytes:
        """
        Get counts.
        :return: Counts.
        """
        return self['counts']  # type: ignore


class Translation(List[float]):
    """Translation Properties."""

    @property
    def x(self) -> float:
        """
        Get the x value.
        :return: The x value.
        """
        return self[0]

    @property
    def y(self) -> float:
        """
        Get the y value.
        :return: The y value.
        """
        return self[1]

    @property
    def z(self) -> float:
        """
        Get the z value.
        :return: The z value.
        """
        return self[2]


class Rotation(List[float]):
    """Rotation Properties."""

    @property
    def w(self) -> float:
        """
        Get the w component of the quaternion.
        :return: The w value.
        """
        return self[0]

    @property
    def x(self) -> float:
        """
        Get the x component of the quaternion.
        :return: The x value.
        """
        return self[1]

    @property
    def y(self) -> float:
        """
        Get the y component of the quaternion.
        :return: The y value.
        """
        return self[2]

    @property
    def z(self) -> float:
        """
        Get the z component of the quaternion.
        :return: The z value.
        """
        return self[3]


class Visibility(Enum):
    """Visibility enumerations."""

    v0_20 = 'v0-20'
    v20_40 = 'v20-40'
    v40_60 = 'v40-60'
    v60_80 = 'v60-80'
    v80_100 = 'v80-100'
    unknown = 'unknown'


class Bbox(List[int]):
    """Bbox Properties."""

    @property
    def xmin(self) -> int:
        """
        Get the minimum x value.
        :return: The minimum x value.
        """
        return self[0]

    @property
    def ymin(self) -> int:
        """
        Get the minimum y value.
        :return: The minimum y value.
        """
        return self[1]

    @property
    def xmax(self) -> int:
        """
        Get the maximum x value.
        :return: The maximum x value.
        """
        return self[2]

    @property
    def ymax(self) -> int:
        """
        Get the maximum y value.
        :return: The maximum y value.
        """
        return self[3]

    @property
    def left(self) -> int:
        """
        Get the left most x value.
        :return: The left most x value.
        """
        return self[0]

    @property
    def top(self) -> int:
        """
        Get the top y value.
        :return: The top y value.
        """
        return self[1]

    @property
    def right(self) -> int:
        """
        Get the right most x value.
        :return: The right most x value.
        """
        return self[2]

    @property
    def bottom(self) -> int:
        """
        Get the bottom y value.
        :return: The bottom y value..
        """
        return self[3]


class Size(List[float]):
    """Size Properties."""

    @property
    def width(self) -> float:
        """
        Get the width.
        :return: The width.
        """
        return self[0]

    @property
    def length(self) -> float:
        """
        Get the length.
        :return: The length.
        """
        return self[1]

    @property
    def height(self) -> float:
        """
        Get the height.
        :return: The height.
        """
        return self[2]


class CameraIntrinsic(List[List[float]]):
    """
    http://ksimek.github.io/2013/08/13/intrinsic/
        [ fx s  px ]
        [ 0  fy py ]
        [ 0  0  1  ]

        fx, fy: focal length
        px, py: principal point offset
        s: axis skew
    """

    @property
    def fx(self) -> float:
        """
        Get the focal length along x.
        :return: The focal length along x.
        """
        return self[0][0]

    @property
    def fy(self) -> float:
        """
        Get the focal length along y.
        :return: The focal length along y.
        """
        return self[1][1]

    @property
    def px(self) -> float:
        """
        Get the principal point offset along x.
        :return: The principal point offset along x.
        """
        return self[0][2]

    @property
    def py(self) -> float:
        """
        Get the principal point offset along y.
        :return: The principal point offset along y.
        """
        return self[1][2]

    @property
    def s(self) -> float:
        """
        Get the axis skew.
        :return: The axis skew.
        """
        return self[0][1]
