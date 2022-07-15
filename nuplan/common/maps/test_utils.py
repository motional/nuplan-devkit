from typing import Any, Dict, List, Optional

import pytest
from shapely.geometry import Polygon

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map_objects import (
    AbstractMapObject,
    GraphEdgeMapObject,
    Intersection,
    PolygonMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def _color_to_object_mapping(layer: SemanticMapLayer) -> List[float]:
    color_mapping = {
        SemanticMapLayer.STOP_LINE: [1.0, 0.0, 0.0, 1.0],
        SemanticMapLayer.CROSSWALK: [0.0, 0.0, 1.0, 1.0],
        SemanticMapLayer.INTERSECTION: [0.0, 1.0, 0.0, 1.0],
        SemanticMapLayer.ROADBLOCK: [0.0, 1.0, 1.0, 1.0],
        SemanticMapLayer.ROADBLOCK_CONNECTOR: [0.0, 1.0, 1.0, 1.0],
    }
    try:
        return color_mapping[layer]
    except KeyError:
        return [1.0, 1.0, 1.0, 0.5]


def add_marker_to_scene(scene: Dict[str, Any], marker_id: str, pose: StateSE2) -> None:
    """
    Serialize and append a marker to the scene.
    :param scene: scene dict.
    :param marker_id: A unique id of the marker.
    :param pose: The pose of the marker.
    """
    if "markers" not in scene.keys():
        scene["markers"] = []

    scene["markers"].append({"id": int(marker_id), "name": marker_id, "pose": pose.serialize(), "shape": "arrow"})


def add_polyline_to_scene(scene: Dict[str, Any], polyline: List[StateSE2]) -> None:
    """
    Serialize and append a polyline to the scene.
    :param scene: scene dict.
    :param polyline: The polyline to be added.
    """
    if "path_info" not in scene.keys():
        scene["path_info"] = []

    scene["path_info"].extend([[pose.x, pose.y, pose.heading] for pose in polyline])


def add_polygon_to_scene(scene: Dict[str, Any], polygon: Polygon, polygon_id: str, color: List[float]) -> None:
    """
    Serialize and append a Polygon to the scene.
    :param scene: scene dict.
    :param polygon: The polygon to be added.
    :param polygon_id: A unique id of the polygon.
    :param color: color of polygon.
    """
    if "shapes" not in scene.keys():
        scene["shapes"] = dict()

    scene["shapes"][str(polygon_id)] = {
        "color": color,
        "filled": True,
        "objects": [[[x, y] for x, y in zip(*polygon.exterior.xy)]],
    }


def add_map_objects_to_scene(
    scene: Dict[str, Any], map_object: List[AbstractMapObject], layer: Optional[SemanticMapLayer] = None
) -> None:
    """
    Serialize and append map objects to the scene.
    :param scene: scene dict.
    :param map_object: The map object to be added.
    :param layer: SemanticMapLayer type.
    """
    for obj in map_object:
        if isinstance(obj, (StopLine, PolygonMapObject, Intersection, RoadBlockGraphEdgeMapObject)):
            add_polygon_to_scene(scene, obj.polygon, obj.id, _color_to_object_mapping(layer))
        elif isinstance(obj, GraphEdgeMapObject):
            add_polyline_to_scene(scene, obj.baseline_path.discrete_path)


def compare_poses(pose1: StateSE2, pose2: StateSE2) -> None:
    """
    Compare x, y, and heading attribute of a StateSE2.
    :param pose1: first pose for comparing.
    :param pose2: second pose for comparing.
    """
    assert pose1.x == pytest.approx(pose2.x, 1e-3)
    assert pose1.y == pytest.approx(pose2.y, 1e-3)
    assert pose1.heading == pytest.approx(pose2.heading, 1e-3)


def compare_map_objects(map_objects_1: List[GraphEdgeMapObject], map_objects_2: List[GraphEdgeMapObject]) -> None:
    """
    Compares two lists of GraphEdgeMapObjects. Note, only the first list will be rendered.
    :param map_objects_1: First list of GraphEdgeMapObjects.
    :param map_objects_2: Second list of GraphEdgeMapObjects.
    """
    assert type(map_objects_1) == type(map_objects_2), (
        f"Map objects are not of the same type." f"Got {type(map_objects_1)} and {type(map_objects_2)}"
    )

    map_object_2_dict = {lc.id: lc for lc in map_objects_2}

    # Check if the same set of connections are found
    assert {lc.id for lc in map_objects_1} == set(map_object_2_dict.keys())

    for map_object_1 in map_objects_1:
        map_object_2 = map_object_2_dict[map_object_1.id]

        # Get discrete baseline path
        blp_1 = map_object_1.baseline_path.discrete_path
        blp_2 = map_object_2.baseline_path.discrete_path

        # Check correctness of baseline paths
        compare_poses(blp_1[0], blp_2[0])
        compare_poses(blp_1[-1], blp_2[-1])
