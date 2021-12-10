from __future__ import annotations

from typing import List, Optional, Tuple, Union

import geopandas as gp
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import box3d_to_polygon
from nuplan.planning.simulation.observation.smart_agents.occupancy_map.abstract_occupancy_map import Geometry, \
    OccupancyMap
from shapely.geometry import LineString, Polygon


class GeoPandasOccupancyMap(OccupancyMap):

    def __init__(self, dataframe: gp.GeoDataFrame):
        self._dataframe = dataframe

    def get_nearest_entry_to(self, geometry_id: str) -> Tuple[str, Geometry, float]:
        """ Inherited, see superclass. """
        assert self.contains(geometry_id), "This occupancy map does not contain given geometry id"

        polygons = self._dataframe
        polygon = self._dataframe.loc[geometry_id]['geometry']

        polygons.drop(geometry_id, axis=0, inplace=True)

        distances = polygons.distance(polygon).sort_values()
        polygon_index = distances.index[0]
        return polygon_index, self._dataframe.loc[polygon_index]["geometry"], distances[0]

    def intersects(self, geometry: Geometry) -> OccupancyMap:
        """ Inherited, see superclass. """
        candidate_df = {'geometry': [geometry]}
        return GeoPandasOccupancyMap(gp.sjoin(self._dataframe,
                                              gp.GeoDataFrame(candidate_df),
                                              how="inner",
                                              op='intersects'))

    def insert(self, geometry_id: str, geometry: Geometry) -> None:
        """ Inherited, see superclass. """
        candidate_df = {'geometry': [geometry]}
        self._dataframe = self._dataframe.append(gp.GeoDataFrame(candidate_df, index=[geometry_id]))

    def get(self, geometry_id: str) -> Geometry:
        """ Inherited, see superclass. """
        return self._dataframe.loc[geometry_id]["geometry"]

    def set(self, geometry_id: str, geometry: Geometry) -> None:
        """ Inherited, see superclass. """
        self._dataframe.loc[geometry_id] = geometry

    def get_all_ids(self) -> List[str]:
        """ Inherited, see superclass. """
        return list(self._dataframe.index)

    def get_all_geometries(self) -> List[Geometry]:
        """ Inherited, see superclass. """
        return list(self._dataframe.geometry)

    @property
    def size(self) -> int:
        """ Inherited, see superclass. """
        index = self._dataframe.index
        return len(index)

    def is_empty(self) -> bool:
        """ Inherited, see superclass. """
        return self._dataframe.empty  # type: ignore

    def contains(self, geometry_id: str) -> bool:
        """ Inherited, see superclass. """
        return geometry_id in self._dataframe.index


class GeoPandasOccupancyMapFactory:

    @staticmethod
    def get_from_boxes(boxes: List[Box3D]) -> OccupancyMap:
        """
        Converts a list of Box3D to a GeopandaDataFrame. The data frame will have the format
           index           geometry
        0  token1          Polygon
        1  token2          Polygon
        The polygon is derived from the corners of each Box3D
        :param boxes: list of Box3D to be converted
        :return: gp.GeoDataFrame
        """
        return GeoPandasOccupancyMap(gp.GeoDataFrame([[box3d_to_polygon(box)] for box in boxes],
                                                     columns=['geometry'],
                                                     geometry='geometry',
                                                     index=[box.token for box in boxes]))

    @staticmethod
    def get_from_geometry(geometries: List[Union[Polygon, LineString]],
                          geometry_ids: Optional[List[str]] = None) -> OccupancyMap:
        """
        Converts a list of shapely.geometry.Polygon to a GeopandaDataFrame. The data frame will have the format
           index           geometry
        0  token1          [Polygon, LineString]
        1  token2          [Polygon, LineString]
        :param geometries: list of [Polygon, LineString]
        :param geometry_ids: list of corresponding ids
        :return: gp.GeoDataFrame
        """

        # Assign some default ids
        if geometry_ids is None:
            geometry_ids = [str(idx) for idx in range(len(geometries))]

        return GeoPandasOccupancyMap(gp.GeoDataFrame([[poly] for poly in geometries],
                                                     columns=['geometry'],
                                                     geometry='geometry',
                                                     index=geometry_ids))
