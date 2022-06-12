from __future__ import annotations

import abc
from typing import Any, Dict


class BoxInterface(abc.ABC):
    """Interface for box."""

    @property  # type: ignore
    @abc.abstractmethod
    def label(self) -> int:
        """
        Label id.
        :return: Label id.
        """
        pass

    @label.setter  # type: ignore
    @abc.abstractmethod
    def label(self, label: int) -> None:
        """
        Sets label id.
        :param label: label id.
        """
        pass

    @property  # type: ignore
    @abc.abstractmethod
    def score(self) -> float:
        """
        Classification score.
        :return: Classification score.
        """
        pass

    @score.setter  # type: ignore
    @abc.abstractmethod
    def score(self, score: float) -> None:
        """
        Sets classification score.
        :param score: Classification score.
        """
        pass

    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """
        Serializes the box instance to a JSON-friendly vector representation.
        :return: Encoding of the box.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> BoxInterface:
        """
        Instantiates a Box3D instance from serialized vector representation.
        :param data: Output from serialize.
        :return: Deserialized box.
        """
        pass
