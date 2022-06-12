from __future__ import annotations

import math
from enum import IntEnum
from functools import cached_property, lru_cache
from typing import List

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


def collision_by_radius_check(box1: OrientedBox, box2: OrientedBox) -> bool:
    """
    Quick check for whether two boxes are in collision using an over-approximated circle around each box
    :param box1: Oriented box (e.g., of ego)
    :param box2: Oriented box (e.g., of other tracks)
    :return False if the distance between centers of the two boxes is larger than a non_overlapping_diagonal
    threshold (circles are external tangents), else True.
    """
    distance_between_centers = box1.center.distance_to(box2.center)
    w1, l1 = box1.width, box1.length
    w2, l2 = box2.width, box2.length
    non_overlapping_diagonal = (np.hypot(w1, l1) + np.hypot(w2, l2)) / 2.0

    return bool(distance_between_centers < non_overlapping_diagonal)


def in_collision(box1: OrientedBox, box2: OrientedBox) -> bool:
    """
    Check for collision between two boxes. First do a quick check by approximating each box with a circle,
    if there is an overlap, check for the exact intersection using geometry Polygon
    :param box1: Oriented box (e.g., of ego)
    :param box2: Oriented box (e.g., of other tracks)
    :return True if there is a collision between the two boxes.
    """
    return bool(box1.geometry.intersects(box2.geometry)) if collision_by_radius_check(box1, box2) else False
