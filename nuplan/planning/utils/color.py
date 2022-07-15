from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from typing import Iterator, List

from numpy import clip


@dataclass(frozen=True)
class SceneColor:
    """
    Represents all colors needed for a scene.
    """

    trajectory_color: Color
    prediction_bike_color: Color
    prediction_pedestrian_color: Color
    prediction_vehicle_color: Color

    def __iter__(self) -> Iterator[Color]:
        """
        Return color components.
        """
        return iter(astuple(self))

    def __mul__(self, other: float) -> SceneColor:
        """
        Return new SceneColor with all color components multiplied by other.
        :param other: Factor to multiply all colors by.
        :return: A new SceneColor, with each color being multiplied by other.
        """
        if isinstance(other, (float, int)):
            return SceneColor(*(color * other for color in self))
        else:
            raise TypeError(f"TypeError: unsupported operand type(s) for *: 'SceneColor' and '{type(other)}')")

    def __rmul__(self, other: float) -> SceneColor:
        """
        Return new SceneColor with all color components multiplied by other.
        :param other: Factor to multiply all colors by.
        :return: A new SceneColor, with each color being multiplied by other.
        """
        return self.__mul__(other)


@dataclass(frozen=True)
class Color:
    """
    Represents a color.
    """

    red: int  # Red component of the color in range 0-255.
    green: int  # Green component of the color in range 0-255.
    blue: int  # Blue component of the color in range 0-255.
    alpha: int  # Alpha channel intensity in range 0-255.

    def __post_init__(self) -> None:
        """
        Checks that the component values are in integers in range 0-255.
        """
        for dim in fields(self):
            component = getattr(self, dim.name)
            if not isinstance(component, int):
                raise TypeError(
                    f"TypeError: Invalid type {type(component)} for color field {dim.name}. Expected type int."
                )
            if component < 0 or component > 255:
                raise ValueError(
                    f"ValueError: Invalid value {component} for color field {dim.name}. Expected value in range 0-255."
                )

    def __iter__(self) -> Iterator[int]:
        """
        Return RGBA components in order red, green, blue, alpha
        """
        return iter(astuple(self))

    def __mul__(self, other: float) -> Color:
        """
        Return a new color with RGBA components multiplied by other. The resulting values are clipped between 0 and 255.
        :param other: Factor to multiply color by.
        :return: A new Color instance with values multiplied by other, clipped between 0 and 255.
        """
        if isinstance(other, (float, int)):
            return Color(*(int(clip(component * other, 0, 255)) for component in self))
        else:
            raise TypeError(f"TypeError: unsupported operand type(s) for *: 'Color' and '{type(other)}')")

    def __rmul__(self, other: float) -> Color:
        """
        Return a new color with RGBA components multiplied by other. The resulting values are clipped between 0 and 255.
        :param other: Factor to multiply color by.
        :return: A new Color instance with values multiplied by other, clipped between 0 and 255.
        """
        return self.__mul__(other)

    def to_float(self) -> List[float]:
        """
        Return RGBA components as a list of floats (0-1).
        :return: list of floats representing red, green, blue and alpha values.
        """
        return [component / 255 for component in self]

    def to_list(self) -> List[int]:
        """
        Return RGBA components as a list of ints (0-255).
        :return: list of floats representing red, green, blue and alpha values.
        """
        return [component for component in self]
