from functools import cached_property
from typing import List, Tuple

from shapely.geometry import Polygon

from nuplan.common.maps.abstract_map_objects import Intersection, Lane, LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import IntersectionType, VectorLayer
from nuplan.common.maps.nuplan_map.utils import get_row_with_value


class NuPlanIntersection(Intersection):
    """
    NuPlanMap implementation of Intersection.
    """

    def __init__(self, intersection_id: str, intersections_df: VectorLayer) -> None:
        """
        Constructor of NuPlanIntersection.
        :param intersection_id: unique identifier of the intersection.
        :param intersections_df: the geopandas GeoDataframe that contains all intersections in the map.
        """
        self._intersections_df = intersections_df
        self._intersection = get_row_with_value(self._intersections_df, "fid", intersection_id)
        super().__init__(intersection_id, IntersectionType.DEFAULT)  # GPKG does not support intersection types yet

    @cached_property
    def polygon(self) -> Polygon:
        """Inherited from superclass."""
        return self._intersection.geometry

    @cached_property
    def interior_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass"""
        raise NotImplementedError

    @cached_property
    def incoming_edges(self) -> List[Lane]:
        """Inherited from superclass"""
        raise NotImplementedError

    @cached_property
    def center(self) -> Tuple[float, float]:
        """
        Returns center of intersection
        :return: Center of intersection
        """
        raise NotImplementedError

    @cached_property
    def is_signaled(self) -> bool:
        """
        Returns if intersection is signaled
        :return: True if intersection is signaled else False
        """
        raise NotImplementedError
