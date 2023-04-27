from functools import cached_property

from shapely.geometry import Polygon

from nuplan.common.maps.abstract_map_objects import Intersection, RoadBlockGraphEdgeMapObject, StopLine
from nuplan.common.maps.maps_datatypes import StopLineType, VectorLayer
from nuplan.common.maps.nuplan_map.utils import get_row_with_value


class NuPlanStopLine(StopLine):
    """
    NuPlanMap implementation of StopLine.
    """

    def __init__(self, stop_line_id: str, stop_lines_df: VectorLayer) -> None:
        """
        Constructor of NuPlanStopLine.
        :param stop_line_id: unique identifier of the stop line.
        :param stop_lines_df: the geopandas GeoDataframe that contains all stop lines in the map.
        """
        self._stop_lines_df = stop_lines_df
        self._stop_line = get_row_with_value(self._stop_lines_df, "fid", stop_line_id)
        super().__init__(stop_line_id, self._stop_line["stop_polygon_type_fid"])

    @cached_property
    def polygon(self) -> Polygon:
        """Inherited from superclass."""
        return self._stop_line.geometry

    @cached_property
    def intersection_from(self) -> Intersection:
        """Inherited from superclass"""
        raise NotImplementedError

    @cached_property
    def layer_type(self) -> StopLineType:
        """Inherited from superclass"""
        raise NotImplementedError

    @cached_property
    def parent(self) -> RoadBlockGraphEdgeMapObject:
        """Inherited from superclass"""
        raise NotImplementedError
