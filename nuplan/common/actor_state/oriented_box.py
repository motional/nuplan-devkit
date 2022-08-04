from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property, lru_cache
from typing import List, Optional

import numpy as np
from shapely.geometry import Polygon

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.transform import translate_longitudinally_and_laterally


class OrientedBoxPointType(IntEnum):
    """Enum for the point of interest in the oriented box."""

    FRONT_BUMPER = (1,)
    REAR_BUMPER = (2,)
    FRONT_LEFT = (3,)
    FRONT_RIGHT = (4,)
    REAR_LEFT = (5,)
    REAR_RIGHT = (6,)
    CENTER = (7,)
    LEFT = (8,)
    RIGHT = 9


@dataclass(frozen=True)
class Dimension:
    """
    Dimensions of an oriented box
    """

    length: float  # [m] dimension
    width: float  # [m] dimension
    height: float  # [m] dimension


class OrientedBox:
    """Represents the physical space occupied by agents on the plane."""

    def __init__(self, center: StateSE2, length: float, width: float, height: float):
        """
        :param center: The pose of the geometrical center of the box
        :param length: The length of the OrientedBox
        :param width: The width of the OrientedBox
        :param height: The height of the OrientedBox
        """
        self._center = center
        self._length = length
        self._width = width
        self._height = height

    @property
    def dimensions(self) -> Dimension:
        """
        :return: Dimensions of this oriented box in meters
        """
        return Dimension(length=self.length, width=self.width, height=self.height)

    @lru_cache()
    def corner(self, point: OrientedBoxPointType) -> Point2D:
        """
        Extract a point of oriented box
        :param point: which point you want to query
        :return: Coordinates of a point on oriented box.
        """
        if point == OrientedBoxPointType.FRONT_LEFT:
            return translate_longitudinally_and_laterally(self.center, self.half_length, self.half_width).point
        elif point == OrientedBoxPointType.FRONT_RIGHT:
            return translate_longitudinally_and_laterally(self.center, self.half_length, -self.half_width).point
        elif point == OrientedBoxPointType.REAR_LEFT:
            return translate_longitudinally_and_laterally(self.center, -self.half_length, self.half_width).point
        elif point == OrientedBoxPointType.REAR_RIGHT:
            return translate_longitudinally_and_laterally(self.center, -self.half_length, -self.half_width).point
        elif point == OrientedBoxPointType.CENTER:
            return self._center.point
        elif point == OrientedBoxPointType.FRONT_BUMPER:
            return translate_longitudinally_and_laterally(self.center, self.half_length, 0.0).point
        elif point == OrientedBoxPointType.REAR_BUMPER:
            return translate_longitudinally_and_laterally(self.center, -self.half_length, 0.0).point
        elif point == OrientedBoxPointType.LEFT:
            return translate_longitudinally_and_laterally(self.center, 0, self.half_width).point
        elif point == OrientedBoxPointType.RIGHT:
            return translate_longitudinally_and_laterally(self.center, 0, -self.half_width).point
        else:
            raise RuntimeError(f"Unknown point: {point}!")

    def all_corners(self) -> List[Point2D]:
        """
        Return 4 corners of oriented box (FL, RL, RR, FR)
        :return: all corners of a oriented box in a list
        """
        return [
            self.corner(OrientedBoxPointType.FRONT_LEFT),
            self.corner(OrientedBoxPointType.REAR_LEFT),
            self.corner(OrientedBoxPointType.REAR_RIGHT),
            self.corner(OrientedBoxPointType.FRONT_RIGHT),
        ]

    @property
    def width(self) -> float:
        """
        Returns the width of the OrientedBox
        :return: The width of the OrientedBox
        """
        return self._width

    @property
    def half_width(self) -> float:
        """
        Returns the half width of the OrientedBox
        :return: The half width of the OrientedBox
        """
        return self._width / 2.0

    @property
    def length(self) -> float:
        """
        Returns the length of the OrientedBox
        :return: The length of the OrientedBox
        """
        return self._length

    @property
    def half_length(self) -> float:
        """
        Returns the half length of the OrientedBox
        :return: The half length of the OrientedBox
        """
        return self._length / 2.0

    @property
    def height(self) -> float:
        """
        Returns the height of the OrientedBox
        :return: The height of the OrientedBox
        """
        return self._height

    @property
    def half_height(self) -> float:
        """
        Returns the half height of the OrientedBox
        :return: The half height of the OrientedBox
        """
        return self._height / 2.0

    @property
    def center(self) -> StateSE2:
        """
        Returns the pose of the center of the OrientedBox
        :return: The pose of the center
        """
        return self._center

    @cached_property
    def geometry(self) -> Polygon:
        """
        Returns the Polygon describing the OrientedBox, if not done yet it will build it lazily.
        :return: The Polygon of the OrientedBox
        """
        corners = [tuple(corner) for corner in self.all_corners()]
        return Polygon(corners)

    def __hash__(self) -> int:
        """
        :return: hash for this object
        """
        return hash((self.center, self.width, self.height, self.length))

    def __eq__(self, other: object) -> bool:
        """
        Compare two oriented boxes
        :param other: object
        :return: true if other and self is equal
        """
        if not isinstance(other, OrientedBox):
            # Return NotImplemented in case the classes are not of the same type
            return NotImplemented
        return (
            math.isclose(self.width, other.width)
            and math.isclose(self.height, other.height)
            and math.isclose(self.length, other.length)
            and self.center == other.center
        )

    @classmethod
    def from_new_pose(cls, box: OrientedBox, pose: StateSE2) -> OrientedBox:
        """
        Initializer that create the same oriented box in a different pose.
        :param box: A sample box
        :param pose: The new pose
        :return: A new OrientedBox
        """
        return cls(pose, box.length, box.width, box.height)


def collision_by_radius_check(box1: OrientedBox, box2: OrientedBox, radius_threshold: Optional[float]) -> bool:
    """
    Quick check for whether two boxes are in collision using a radius check, if radius_threshold is None, an over-approximated circle around each box is considered to determine the radius
    :param box1: Oriented box (e.g., of ego)
    :param box2: Oriented box (e.g., of other tracks)
    :param radius_threshold: Radius threshold for quick collision check
    :return False if the distance between centers of the two boxes is larger than radius_threshold else True. If radius_threshold is None, radius_threshold is defined as the sum of the radius of the smallest over-approximated circles around each box
    centered at the box center (i.e., the radius_threshold is defined when over-approximated circles are external tangents).
    """
    if not radius_threshold:
        w1, l1 = box1.width, box1.length
        w2, l2 = box2.width, box2.length
        radius_threshold = (np.hypot(w1, l1) + np.hypot(w2, l2)) / 2.0

    distance_between_centers = box1.center.distance_to(box2.center)

    return bool(distance_between_centers < radius_threshold)


def in_collision(box1: OrientedBox, box2: OrientedBox, radius_threshold: Optional[float] = None) -> bool:
    """
    Check for collision between two boxes. First do a quick check by approximating each box with a circle of given radius,
    if there is an overlap, check for the exact intersection using geometry Polygon
    :param box1: Oriented box (e.g., of ego)
    :param box2: Oriented box (e.g., of other tracks)
    :param radius: Radius for quick collision check
    :return True if there is a collision between the two boxes.
    """
    return (
        bool(box1.geometry.intersects(box2.geometry))
        if collision_by_radius_check(box1, box2, radius_threshold)
        else False
    )
