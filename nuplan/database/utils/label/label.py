from __future__ import annotations

from typing import Any, Dict, Tuple

Color = Tuple[int, int, int, int]


class Label:
    """A label with the name and color."""

    def __init__(self, name: str, color: Color) -> None:
        """
        :param name: The name of the color.
        :param color: An R, G, B, alpha tuple which defines the color.
        """
        self.name = name
        self.color = color

        for c in self.color:
            assert 0 <= c <= 255

    def __repr__(self) -> str:
        """
        Represents a label using a string.
        :return: A string to represent a label.
        """
        return "Label(name='{}', color={})".format(self.name, self.color)

    def __eq__(self, other: object) -> bool:
        """
        Checks if two labels are equal.
        :param other: Other object.
        :return: True if both objects are the same.
        """
        if not isinstance(other, Label):
            return NotImplemented

        return self.name == other.name and self.color == other.color

    @property
    def normalized_color(self) -> Tuple[float, ...]:
        """
        Normalized color used for pyplot.
        :return: Normalized color.
        """
        return tuple(c / 255.0 for c in self.color)

    def serialize(self) -> Dict[str, Any]:
        """
        Serializes the label instance to a JSON-friendly dictionary representation.
        :return: Encoding of the label.
        """
        return {'name': self.name, 'color': self.color}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Label:
        """
        Instantiates a Label instance from serialized dictionary representation.
        :param data: Output from serialize.
        :return: Deserialized label.
        """
        return Label(name=data['name'], color=tuple(int(channel) for channel in data['color']))  # type: ignore
