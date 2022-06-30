from functools import cached_property

import pandas as pd
from shapely.geometry import Polygon

from nuplan.common.maps.abstract_map_objects import PolygonMapObject
from nuplan.common.maps.maps_datatypes import VectorLayer
from nuplan.common.maps.nuplan_map.utils import get_row_with_value


class NuPlanPolygonMapObject(PolygonMapObject):
    """
    NuPlanMap implementation of Polygon Map Object.
    """

    def __init__(self, generic_polygon_area_id: str, generic_polygon_area: VectorLayer):
        """
        Constructor of generic polygon map layer.
        This includes:
            - CROSSWALK
            - WALKWAYS
            - CARPARK_AREA
            - PUDO
        :param generic_polygon_area_id: Generic polygon area id.
        :param generic_polygon_area: Generic polygon area.
        """
        super().__init__(generic_polygon_area_id)
        self._generic_polygon_area = generic_polygon_area
        self._area = None

    @cached_property
    def polygon(self) -> Polygon:
        """Inherited from superclass."""
        return self._get_area().geometry

    def _get_area(self) -> pd.Series:
        """
        Gets the series from the polygon dataframe containing polygon's id.
        :return: The respective series from the polygon dataframe.
        """
        if self._area is None:
            self._area = get_row_with_value(self._generic_polygon_area, "fid", self.id)

        return self._area
