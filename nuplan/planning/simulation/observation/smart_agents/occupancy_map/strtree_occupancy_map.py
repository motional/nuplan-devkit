from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import box3d_to_polygon
from nuplan.planning.simulation.observation.smart_agents.occupancy_map.abstract_occupancy_map import Geometry, \
    OccupancyMap
from shapely.ops import nearest_points
from shapely.strtree import STRtree

GeometryMap = Dict[str, Geometry]


class SRTreeOccupancyMap(OccupancyMap):

    def __init__(self, geom_map: GeometryMap):
        self._geom_map: GeometryMap = geom_map

    def get_nearest_entry_to(self, geometry_id: str) -> Tuple[str, Geometry, float]:
        """ Inherited, see superclass. """

        assert self.contains(geometry_id), "This occupancy map does not contain given geometry id"

        strtree, index_by_id = self._build_strtree(geometry_id)
        nearest = strtree.nearest(self.get(geometry_id))
        p1, p2 = nearest_points(self.get(geometry_id), nearest)
        return index_by_id[id(nearest)], nearest, p1.distance(p2)

    def intersects(self, geometry: Geometry) -> OccupancyMap:
        """ Inherited, see superclass. """
        strtree, index_by_id = self._build_strtree()
        return SRTreeOccupancyMap({index_by_id[id(geom)]: geom for geom in strtree.query(geometry)
                                   if geom.intersects(geometry)})

    def insert(self, geometry_id: str, geometry: Geometry) -> None:
        """ Inherited, see superclass. """
        self._geom_map[geometry_id] = geometry

    def get(self, geometry_id: str) -> Geometry:
        """ Inherited, see superclass. """
        return self._geom_map[geometry_id]

    def set(self, geometry_id: str, geometry: Geometry) -> None:
        """ Inherited, see superclass. """
        self._geom_map[geometry_id] = geometry

    def get_all_ids(self) -> List[str]:
        """ Inherited, see superclass. """
        return list(self._geom_map.keys())

    def get_all_geometries(self) -> List[Geometry]:
        """ Inherited, see superclass. """
        return list(self._geom_map.values())

    @property
    def size(self) -> int:
        """ Inherited, see superclass. """
        return len(self._geom_map)

    def is_empty(self) -> bool:
        """ Inherited, see superclass. """
        return not self._geom_map

    def contains(self, geometry_id: str) -> bool:
        """ Inherited, see superclass. """
        return geometry_id in self._geom_map

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

        strtree = STRtree(temp_geom_map.values())
        index_by_id = {id(geom): geom_id for geom_id, geom in temp_geom_map.items()}

        return strtree, index_by_id


class STRTreeOccupancyMapFactory:

    @staticmethod
    def get_from_boxes(boxes: List[Box3D]) -> OccupancyMap:
        """
        Builds an STRTreeOccupancyMap from a list of Box3D. The underlying dictionary will have the format
          key    : value
          token1 : [Polygon, LineString]
          token2 : [Polygon, LineString]
        The polygon is derived from the corners of each Box3D
        :param boxes: list of Box3D to be converted
        :return: STRTreeOccupancyMap
        """
        return SRTreeOccupancyMap({box.token: box3d_to_polygon(box) for box in boxes if box.token is not None})

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
            return SRTreeOccupancyMap({str(geom_id): geom for geom_id, geom in enumerate(geometries)})

        return SRTreeOccupancyMap({str(geom_id): geom for geom_id, geom in zip(geometry_ids, geometries)})
