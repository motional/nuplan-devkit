from nuplan.common.maps.abstract_map_objects import Intersection
from nuplan.common.maps.maps_datatypes import IntersectionType, VectorLayer
from nuplan.common.maps.nuplan_map.utils import get_element_with_fid
from shapely.geometry import Polygon


class NuPlanIntersection(Intersection):
    def __init__(self, intersection_id: str, intersections_df: VectorLayer) -> None:
        self._intersections_df = intersections_df
        self._intersection = get_element_with_fid(self._intersections_df, "fid", intersection_id)
        super().__init__(intersection_id, IntersectionType.DEFAULT)  # GPKG does not support intersection types yet

    @property
    def polygon(self) -> Polygon:
        """ Inherited from superclass """
        return self._intersection.geometry
