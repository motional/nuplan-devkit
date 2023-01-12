from __future__ import annotations

from typing import List, Optional, Tuple, Union

import geopandas as gp
import pandas
from shapely.geometry import LineString, Polygon

from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import Geometry, OccupancyMap


class GeoPandasOccupancyMap(OccupancyMap):
    """OccupancyMap supported by GeoPandas."""

    def __init__(self, dataframe: gp.GeoDataFrame):
        """
        Constructor of GeoPandasOccupancyMap.
        :param dataframe: underlying geopandas dataframe.
        """
        self._dataframe = dataframe

    def get_nearest_entry_to(self, geometry_id: str) -> Tuple[str, Geometry, float]:
        """Inherited, see superclass."""
        assert self.contains(geometry_id), "This occupancy map does not contain given geometry id"

        polygons = self._dataframe
        polygon = self._dataframe.loc[geometry_id]['geometry']

        polygons.drop(geometry_id, axis=0, inplace=True)

        distances = polygons.distance(polygon).sort_values()
        polygon_index = distances.index[0]
        return polygon_index, self._dataframe.loc[polygon_index]["geometry"], distances[0]

    def intersects(self, geometry: Geometry) -> OccupancyMap:
        """Inherited, see superclass."""
        candidate_df = {'geometry': [geometry]}
        return GeoPandasOccupancyMap(
            gp.sjoin(self._dataframe, gp.GeoDataFrame(candidate_df), how="inner", predicate='intersects')
        )

    def insert(self, geometry_id: str, geometry: Geometry) -> None:
        """Inherited, see superclass."""
        candidate_df = {'geometry': [geometry]}
        self._dataframe = pandas.concat([self._dataframe, gp.GeoDataFrame(candidate_df, index=[geometry_id])])

    def get(self, geometry_id: str) -> Geometry:
        """Inherited, see superclass."""
        return self._dataframe.loc[geometry_id]["geometry"]

    def set(self, geometry_id: str, geometry: Geometry) -> None:
        """Inherited, see superclass."""
        self._dataframe.loc[geometry_id] = geometry

    def get_all_ids(self) -> List[str]:
        """Inherited, see superclass."""
        return list(self._dataframe.index)

    def get_all_geometries(self) -> List[Geometry]:
        """Inherited, see superclass."""
        return list(self._dataframe.geometry)

    @property
    def size(self) -> int:
        """Inherited, see superclass."""
        index = self._dataframe.index
        return len(index)

    def is_empty(self) -> bool:
        """Inherited, see superclass."""
        return self._dataframe.empty  # type: ignore

    def contains(self, geometry_id: str) -> bool:
        """Inherited, see superclass."""
        return geometry_id in self._dataframe.index

    def remove(self, geometry_id: List[str]) -> None:
        """Inherited, see superclass."""
        self._dataframe.drop(geometry_id)


class GeoPandasOccupancyMapFactory:
    """Factory for constructing GeoPandasOccupancyMaps."""

    @staticmethod
    def get_from_geometry(
        geometries: List[Union[Polygon, LineString]], geometry_ids: Optional[List[str]] = None
    ) -> OccupancyMap:
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

        return GeoPandasOccupancyMap(
            gp.GeoDataFrame(
                [[poly] for poly in geometries], columns=['geometry'], geometry='geometry', index=geometry_ids
            )
        )
