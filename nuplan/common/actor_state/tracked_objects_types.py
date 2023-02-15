from __future__ import annotations

from enum import Enum
from typing import Set


class TrackedObjectType(Enum):
    """Enum of classification types for TrackedObject."""

    VEHICLE = 0, 'vehicle'
    PEDESTRIAN = 1, 'pedestrian'
    BICYCLE = 2, 'bicycle'
    TRAFFIC_CONE = 3, 'traffic_cone'
    BARRIER = 4, 'barrier'
    CZONE_SIGN = 5, 'czone_sign'
    GENERIC_OBJECT = 6, 'generic_object'
    EGO = 7, 'ego'

    def __int__(self) -> int:
        """
        Convert an element to int
        :return: int
        """
        return self.value  # type: ignore

    def __new__(cls, value: int, name: str) -> TrackedObjectType:
        """
        Create new element
        :param value: its value
        :param name: its name
        """
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name  # type: ignore
        return member

    def __eq__(self, other: object) -> bool:
        """
        Equality checking
        :return: int
        """
        # Cannot check with isisntance, as some code imports this in a different way
        try:
            return self.name == other.name and self.value == other.value  # type: ignore
        except AttributeError:
            return NotImplemented

    def __hash__(self) -> int:
        """Hash"""
        return hash((self.name, self.value))


AGENT_TYPES: Set[TrackedObjectType] = {
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
    TrackedObjectType.EGO,
}

STATIC_OBJECT_TYPES: Set[TrackedObjectType] = {
    TrackedObjectType.CZONE_SIGN,
    TrackedObjectType.BARRIER,
    TrackedObjectType.TRAFFIC_CONE,
    TrackedObjectType.GENERIC_OBJECT,
}
