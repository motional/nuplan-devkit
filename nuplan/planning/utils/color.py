from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from enum import Enum
from typing import Iterator, List

from numpy import clip, integer


class ColorType(Enum):
    """
    Enum class for Color type
    """

    FLOAT = 1
    INT = 255


@dataclass(frozen=True)
class Color:
    """
    Represents a color.
    """

    red: float  # Red component of the color in range 0-1.
    green: float  # Green component of the color in range 0-1.
    blue: float  # Blue component of the color in range 0-1.
    alpha: float  # Alpha channel intensity in range 0-1.
    serialize_to: ColorType = ColorType.FLOAT  # Whether to serialize to 0-1 or 0-255.

    def __post_init__(self) -> None:
        """
        Checks that the component values are floats in range 0-1.
        """
        for dim in fields(self)[:4]:
            component = getattr(self, dim.name)
            if not isinstance(component, (float, int, integer)):
                raise TypeError(
                    f"TypeError: Invalid type {type(component)} for color field {dim.name}. Expected type float."
                )
            if component < 0.0 or component > 1.0:
                raise ValueError(
                    f"ValueError: Invalid value {component} for color field {dim.name}. Expected value in range 0.0-1.0."
                )

    def _serialize(self, value: float) -> float:
        """
        Converts the components into correct value before serializing
        """
        if self.serialize_to == ColorType.INT:
            return int(value * 255)
        else:
            return value

    def __iter__(self) -> Iterator[float]:
        """
        Return RGBA components in order red, green, blue, alpha
        """
        return iter(self._serialize(x) for x in astuple(self)[:4])

    def __mul__(self, other: float) -> Color:
        """
        Return a new color with RGBA components multiplied by other. The resulting values are clipped between 0 and 1.
        :param other: Factor to multiply color by.
        :return: A new Color instance with values multiplied by other, clipped between 0 and 1.
        """
        if isinstance(other, (float, int)):
            return Color(*([clip(component * other, 0, 1) for component in astuple(self)[:4]] + [self.serialize_to]))
        else:
            raise TypeError(f"TypeError: unsupported operand type(s) for *: 'Color' and '{type(other)}')")

    def __rmul__(self, other: float) -> Color:
        """
        Return a new color with RGBA components multiplied by other. The resulting values are clipped between 0 and 1.
        :param other: Factor to multiply color by.
        :return: A new Color instance with values multiplied by other, clipped between 0 and 1.
        """
        return self.__mul__(other)

    def to_list(self) -> List[float]:
        """
        Return RGBA components as a list of ints/floats depending on the value of serialize_to.
        :return: list of floats representing red, green, blue and alpha values.
        """
        return [component for component in self]


@dataclass
class TrajectoryColors:
    """Colors to use for each trajectory in the serialization."""

    ego_predicted_trajectory = Color(0, 0, 1, 0.4, ColorType.INT)  # (r, g, b, a) color
    ego_expert_trajectory = Color(1, 0, 0, 0.4, ColorType.INT)  # (r, g, b, a) color
    agents_predicted_trajectory = Color(0, 1, 0, 0.2, ColorType.INT)  # (r, g, b, a) color


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


StandardSceneColor = SceneColor(
    trajectory_color=Color(0, 1, 0, 0.75, ColorType.INT),
    prediction_bike_color=Color(1, 0, 0, 1, ColorType.FLOAT),
    prediction_pedestrian_color=Color(0, 0, 1, 1, ColorType.FLOAT),
    prediction_vehicle_color=Color(0, 1, 0, 1, ColorType.FLOAT),
)

AltSceneColor = SceneColor(
    trajectory_color=Color(1, 0, 0, 0.75, ColorType.INT),
    prediction_bike_color=Color(1, 0, 0.75, 0.6, ColorType.FLOAT),
    prediction_pedestrian_color=Color(0, 0.75, 1, 0.6, ColorType.FLOAT),
    prediction_vehicle_color=Color(0.75, 1, 0, 0.6, ColorType.FLOAT),
)
