from enum import Enum, auto
from typing import Any, Dict, TextIO, Tuple

import numpy as np
import numpy.typing as npt


def jsontabledump(f: TextIO, c: Tuple[str, Dict[str, Tuple[str, str]]], name: str) -> None:
    """
    Dump table schema to the given file.
    :param f: File object to dump the table to.
    :param c: Table schema.
    :param name: Table name.
    """
    f.write("{}\n---------\n".format(name))
    f.write("\n" + c[0] + "\n")
    f.write("```\n")
    f.write('{}{}\n'.format(name, '{'))
    for key in c[1].keys():
        f.write('   {:27}{} -- {}\n'.format('"' + key + '":', c[1][key][0], c[1][key][1]))
    f.write("{}\n```\n".format('}'))


class MotionalColor(Enum):
    """Color Mappings."""

    BLACK = auto()
    WHITE = auto()

    NEON_YELLOW = auto()
    YELLOW_LIME = auto()
    SELECTIVE_YELLOW = auto()
    TANGERGINE_YELLOW = auto()

    TAUPE = auto()

    PURE_RED = auto()
    PURE_RED_LIGHT = auto()
    RADICAL_RED = auto()
    TOMATO = auto()
    CORAL = auto()
    LIGHT_CORAL = auto()

    PUMPKIN = auto()
    ORANGE = auto()
    SAFETY_ORANGE = auto()
    ORANGE_PEEL = auto()
    ORANGE_RED = auto()
    DARK_ORANGE = auto()

    PURE_BLUE = auto()
    BLUE = auto()
    TEAL = auto()
    AQUA = auto()
    INDIGO = auto()
    TURQOISE = auto()
    CORN_FLOWER_BLUE = auto()
    NAVY = auto()

    PURE_GREEN = auto()
    SPRING_GREEN = auto()
    GREEN = auto()
    OLIVE = auto()

    MAGENTA = auto()
    IMPURE_MAGENTA = auto()
    ELECTRIC_VIOLET = auto()
    BLUE_VIOLET = auto()
    BRILLIANT_ROSE = auto()
    BRIGHT_PINK = auto()
    COSMOS_PINK = auto()

    ROSY_BROWN = auto()
    BURLY_WOOD = auto()

    # Motional color palette
    HYUNDAI_BLUE = auto()
    MOTIONAL_PURPLE = auto()
    LAVENDER_GRAY = auto()
    SEAGLASS = auto()
    SOLID_AQUA = auto()
    LIGHT_ASPHALT = auto()
    MEDIUM_ASPHALT = auto()
    DARK_ASPHALT = auto()
    CHART_YELLOW = auto()
    CHART_GREEN = auto()

    def to_rgb_tuple(self) -> Tuple[int, int, int]:
        """
        Get the RGB color tuple.
        :return: The RGB color tuple.
        """
        rgb_tuples = {
            MotionalColor.PURE_RED: (255, 0, 0),
            MotionalColor.PURE_GREEN: (0, 255, 0),
            MotionalColor.PURE_BLUE: (0, 0, 255),
            MotionalColor.RADICAL_RED: (255, 61, 99),
            MotionalColor.PURE_RED_LIGHT: (255, 0, 0),
            MotionalColor.PUMPKIN: (255, 158, 0),
            MotionalColor.ORANGE: (255, 145, 0),
            MotionalColor.BLUE: (0, 0, 230),
            MotionalColor.SPRING_GREEN: (0, 230, 118),
            MotionalColor.BLACK: (0, 0, 0),
            MotionalColor.WHITE: (255, 255, 255),
            MotionalColor.MAGENTA: (255, 0, 255),
            MotionalColor.IMPURE_MAGENTA: (255, 0, 255),
            MotionalColor.ELECTRIC_VIOLET: (213, 0, 249),
            MotionalColor.NEON_YELLOW: (255, 255, 0),
            MotionalColor.AQUA: (0, 255, 255),
            MotionalColor.TEAL: (0, 207, 191),
            MotionalColor.BRILLIANT_ROSE: (255, 105, 180),
            MotionalColor.INDIGO: (75, 0, 130),
            MotionalColor.TURQOISE: (64, 224, 208),
            MotionalColor.TAUPE: (135, 121, 78),
            MotionalColor.SAFETY_ORANGE: (255, 118, 0),
            MotionalColor.ORANGE_PEEL: (255, 138, 0),
            MotionalColor.SELECTIVE_YELLOW: (255, 178, 0),
            MotionalColor.TANGERGINE_YELLOW: (255, 198, 0),
            MotionalColor.BRIGHT_PINK: (255, 61, 149),
            MotionalColor.COSMOS_PINK: (255, 192, 203),
            MotionalColor.HYUNDAI_BLUE: (10, 114, 41),
            MotionalColor.MOTIONAL_PURPLE: (92, 246, 72),
            MotionalColor.LAVENDER_GRAY: (173, 227, 185),
            MotionalColor.SEAGLASS: (201, 241, 234),
            MotionalColor.SOLID_AQUA: (0, 255, 246),
            MotionalColor.LIGHT_ASPHALT: (210, 220, 220),
            MotionalColor.MEDIUM_ASPHALT: (130, 145, 135),
            MotionalColor.DARK_ASPHALT: (64, 74, 68),
            MotionalColor.CHART_YELLOW: (226, 26, 255),
            MotionalColor.CHART_GREEN: (0, 19, 255),
            MotionalColor.YELLOW_LIME: (244, 255, 0),
            MotionalColor.TOMATO: (255, 71, 99),
            MotionalColor.CORAL: (255, 80, 127),
            MotionalColor.LIGHT_CORAL: (240, 128, 128),
            MotionalColor.ORANGE_RED: (255, 0, 69),
            MotionalColor.DARK_ORANGE: (255, 0, 140),
            MotionalColor.CORN_FLOWER_BLUE: (100, 237, 149),
            MotionalColor.NAVY: (0, 128, 0),
            MotionalColor.GREEN: (0, 0, 128),
            MotionalColor.OLIVE: (128, 0, 128),
            MotionalColor.BLUE_VIOLET: (138, 226, 43),
            MotionalColor.ROSY_BROWN: (188, 143, 143),
            MotionalColor.BURLY_WOOD: (222, 135, 184),
        }
        return rgb_tuples[self]

    def to_rgba_tuple(self, alpha: int = 0) -> Tuple[int, int, int, int]:
        """
        Get the RGBA color tuple.
        :param alpha: Alpha value to append to RGB values, defaults to zero.
        :return: RGBA color tuple.
        """
        return self.to_rgb_tuple() + (alpha,)


def default_color(category_name: str) -> Tuple[int, int, int]:
    """
    Get the default color for a category.

    :param category_name: Category name.
    :return: Default RGB color tuple.
    """
    if 'cycle' in category_name:
        return MotionalColor.RADICAL_RED.to_rgb_tuple()
    elif 'vehicle' in category_name:
        return MotionalColor.PUMPKIN.to_rgb_tuple()
    elif 'human.pedestrian' in category_name:
        return MotionalColor.BLUE.to_rgb_tuple()
    elif 'cone' in category_name or 'barrier' in category_name:
        return MotionalColor.BLACK.to_rgb_tuple()
    # These rest are the same in Scale's annotation tool.
    elif category_name == 'flat.driveable_surface':
        return MotionalColor.ORANGE.to_rgb_tuple()
    elif category_name == 'flat':
        return MotionalColor.SPRING_GREEN.to_rgb_tuple()
    elif category_name == 'vehicle.ego':
        return MotionalColor.ELECTRIC_VIOLET.to_rgb_tuple()
    else:
        return MotionalColor.MAGENTA.to_rgb_tuple()


def default_color_np(category_name: str) -> npt.NDArray[np.float64]:
    """
    Get the default color for a category in numpy.

    :param category_name: Category name.
    :return: <np.float: 3> RGB color.
    """
    return np.array(default_color(category_name)) / 255.0


def simple_repr(record: Any) -> str:
    """
    Simple renderer for a SQL table
    :param record: A table record.
    :return: A string description of the record.
    """
    out = '{:28}: {}\n'.format('token', record.token)
    columns = None
    # get the columns of the table, we'd like to print out only these fields because
    # jupyter notebook sometimes messes them with back_populate fields.
    if hasattr(record, '__table__'):
        columns = {c for c in record.__table__.columns.keys()}

    for field, value in vars(record).items():
        if columns and field not in columns:
            continue
        if not (field[0] == '_' or field == 'token'):
            out += '{:28}: {}\n'.format(field, value)
    return out + '\n'
