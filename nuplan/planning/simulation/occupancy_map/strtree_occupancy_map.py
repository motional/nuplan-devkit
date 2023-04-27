from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from shapely.ops import nearest_points
from shapely.strtree import STRtree

from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import Geometry, OccupancyMap

GeometryMap = Dict[str, Geometry]


class STRTreeOccupancyMap(OccupancyMap):
    """
    OccupancyMap using an SR-tree to support efficient get-nearest queries.
    """

    def __init__(self, geom_map: GeometryMap):
        """
        Constructor of STRTreeOccupancyMap.
        :param geom_map: underlying geometries for occupancy map.
        """
        self._geom_map: GeometryMap = geom_map

    def get_nearest_entry_to(self, geometry_id: str) -> Tuple[str, Geometry, float]:
        """Inherited, see superclass."""
        assert self.contains(geometry_id), "This occupancy map does not contain given geometry id"

        strtree, index_by_id = self._build_strtree(geometry_id)
        nearest_index = strtree.nearest(self.get(geometry_id))
        nearest = strtree.geometries.take(nearest_index)
        p1, p2 = nearest_points(self.get(geometry_id), nearest)
        return index_by_id[id(nearest)], nearest, p1.distance(p2)

    def intersects(self, geometry: Geometry) -> OccupancyMap:
        """Inherited, see superclass."""
        strtree, index_by_id = self._build_strtree()
        indices = strtree.query(geometry)
        return STRTreeOccupancyMap(
            {index_by_id[id(geom)]: geom for geom in strtree.geometries.take(indices) if geom.intersects(geometry)}
        )

    def insert(self, geometry_id: str, geometry: Geometry) -> None:
        """Inherited, see superclass."""
        self._geom_map[geometry_id] = geometry

    def get(self, geometry_id: str) -> Geometry:
        """Inherited, see superclass."""
        return self._geom_map[geometry_id]

    def set(self, geometry_id: str, geometry: Geometry) -> None:
        """Inherited, see superclass."""
        self._geom_map[geometry_id] = geometry

    def get_all_ids(self) -> List[str]:
        """Inherited, see superclass."""
        return list(self._geom_map.keys())

    def get_all_geometries(self) -> List[Geometry]:
        """Inherited, see superclass."""
        return list(self._geom_map.values())

    @property
    def size(self) -> int:
        """Inherited, see superclass."""
        return len(self._geom_map)

    def is_empty(self) -> bool:
        """Inherited, see superclass."""
        return not self._geom_map

    def contains(self, geometry_id: str) -> bool:
        """Inherited, see superclass."""
        return geometry_id in self._geom_map

    def remove(self, geometry_ids: List[str]) -> None:
        """Remove geometries from the occupancy map by ids."""
        for id in geometry_ids:
            assert id in self._geom_map, "Geometry does not exist in occupancy map"
            self._geom_map.pop(id)

    def _get_other_geometries(self, ignore_id: str) -> GeometryMap:
        """
        Returns all geometries as except for one specified by ignore_id

        :param ignore_id: the key corresponding to the geometry to be skipped
        :return: GeometryMap
        """
        return {geom_id: geom for geom_id, geom in self._geom_map.items() if geom_id not in ignore_id}

    def _build_strtree(self, ignore_id: Optional[str] = None) -> Tuple[STRtree, Dict[int, str]]:
        """
        Constructs an STRTree from the geometries stored in the geometry map. Additionally, returns a index-id
        mapping to the original keys of the geometries. Has the option to build a tree omitting on geometry
        :param ignore_id: the key corresponding to the geometry to be skipped
        :return: STRTree containing the values of _geom_map, index mapping to the original keys
        """
        if ignore_id is not None:
            temp_geom_map = self._get_other_geometries(ignore_id)
        else:
            temp_geom_map = self._geom_map

        strtree = STRtree(list(temp_geom_map.values()))
        index_by_id = {id(geom): geom_id for geom_id, geom in temp_geom_map.items()}

        return strtree, index_by_id


class STRTreeOccupancyMapFactory:
    """
    Factory for STRTreeOccupancyMap.
    """

    @staticmethod
    def get_from_boxes(scene_objects: List[SceneObject]) -> OccupancyMap:
        """
        Builds an STRTreeOccupancyMap from a list of SceneObject. The underlying dictionary will have the format
          key    : value
        return {geom_id: geom for geom_id, geom in self._geom_map.items() if ge
          token1 : [Polygon, LineString]
          token2 : [Polygon, LineString]
        The polygon is derived from the corners of each SceneObject
        :param scene_objects: list of SceneObject to be converted
        :return: STRTreeOccupancyMap
        """
        return STRTreeOccupancyMap(
            {
                scene_object.track_token: scene_object.box.geometry
                for scene_object in scene_objects
                if scene_object.track_token is not None
            }
        )

    @staticmethod
    def get_from_geometry(geometries: List[Geometry], geometry_ids: Optional[List[str]] = None) -> OccupancyMap:
        """
        Builds an STRTreeOccupancyMap from a list of Geometry. The underlying dictionary will have the format
          key    : value
          token1 : [Polygon, LineString]
          token2 : [Polygon, LineString]]
        :param geometries: list of [Polygon, LineString]
        :param geometry_ids: list of corresponding ids
        :return: STRTreeOccupancyMap
        """
        if geometry_ids is None:
            return STRTreeOccupancyMap({str(geom_id): geom for geom_id, geom in enumerate(geometries)})

        return STRTreeOccupancyMap({str(geom_id): geom for geom_id, geom in zip(geometry_ids, geometries)})
