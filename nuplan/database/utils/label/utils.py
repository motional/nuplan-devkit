from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Dict, Tuple

from nuplan.database.utils.label.label import Label


class LABEL_MAP(IntEnum):
    """
    Mapping of database's box labels to integers.
    """

    VEHICLE = 1
    BICYCLE = auto()
    PEDESTRIAN = auto()
    TRAFFIC_CONE = auto()
    BARRIER = auto()
    CZONE_SIGN = auto()
    GENERIC_OBJECT = auto()


@dataclass
class LabelMapping:
    """
    Label mapping data class.
    TODO: Temporarily used only for compatibility with agent_state, to be removed!
    """

    local2id: Dict[str, int]


def parse_labelmap_dataclass(
    labelmap: Dict[int, Label]
) -> Tuple[OrderedDict[int, Any], OrderedDict[int, Tuple[Any, ...]]]:
    """
    A labelmap provides a map from integer ids to text and color labels. After loading a label map from json, this
    will parse the labelmap into commonly utilized mappings and fix the formatting issues caused by json.
    :param labelmap: Dictionary of label id and its corresponding Label class information.
    :return: (id2name {id <int>: name <str>}, id2color {id <int>: color (R <int>, G <int>, B <int>, A <int>)}.
        Label id to name and label id to color mappings tuple.
    """
    id2name = OrderedDict()  # {integer ids : str name}
    id2color = OrderedDict()  # {integer ids: RGB or RGBA tuple}

    # Some evaluation tools (like the confusion matrix) need sorted names
    ids = [int(_id) for _id in labelmap.keys()]
    ids.sort()

    for _id in ids:
        id2name[_id] = labelmap[_id].name
        id2color[_id] = tuple(labelmap[_id].color)

    return id2name, id2color


global2local = {
    "generic_object": "generic_object",
    "vehicle": "car",
    "pedestrian": "ped",
    "bicycle": "bike",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
    "czone_sign": "czone_sign",
}

raw_mapping: Dict[str, Dict[Any, Any]] = {
    "global2local": global2local,
    "local2id": {"generic_object": 0, "car": 1, "ped": 2, "bike": 3, "traffic_cone": 4, "barrier": 5, "czone_sign": 6},
    "id2local": {0: "generic_object", 1: "car", 2: "ped", 3: "bike", 4: "traffic_cone", 5: "barrier", 6: "czone_sign"},
    "id2color": {
        0: (0, 255, 0, 0),  # green
        1: (255, 158, 0, 0),  # orange
        2: (0, 0, 250, 0),  # blue
        3: (255, 61, 99, 0),  # red
        4: (0, 0, 0, 0),  # black
        5: (244, 255, 0, 0),  # yellow
        6: (213, 0, 249, 0),  # electric violet
    },
}

PBVTB_LABELMAPPING = LabelMapping(local2id=raw_mapping['local2id'])

local2agent_type = {
    "generic_object": "GENERIC_OBJECT",
    "genericobjects": "GENERIC_OBJECT",
    "obstacles": "GENERIC_OBJECT",
    "car": "VEHICLE",
    "ped": "PEDESTRIAN",
    "bike": "BICYCLE",
    "traffic_cone": "TRAFFIC_CONE",
    "trafficcone": "TRAFFIC_CONE",
    "barrier": "BARRIER",
    "czone_sign": "CZONE_SIGN",
    "czone_signs": "CZONE_SIGN",
    "short_vehicle": "VEHICLE",
    "long_vehicle": "VEHICLE",
}
