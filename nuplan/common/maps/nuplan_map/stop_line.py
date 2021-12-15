from nuplan.common.maps.abstract_map_objects import StopLine
from nuplan.common.maps.maps_datatypes import VectorLayer
from nuplan.common.maps.nuplan_map.utils import get_element_with_fid
from shapely.geometry import Polygon


class NuPlanStopLine(StopLine):
    def __init__(self, stop_line_id: str, stop_lines_df: VectorLayer) -> None:
        self._stop_lines_df = stop_lines_df
        self._stop_line = get_element_with_fid(self._stop_lines_df, "fid", stop_line_id)
        super().__init__(stop_line_id, self._stop_line["stop_polygon_type_fid"])

    @property
    def polygon(self) -> Polygon:
        """ Inherited from superclass """
        return self._stop_line.geometry
