from typing import List, Optional, cast

import pandas as pd
from shapely.geometry import CAP_STYLE, Polygon

import nuplan.common.maps.nuplan_map.lane as lane
from nuplan.common.maps.abstract_map_objects import BaselinePath, GraphEdgeMapObject, LaneConnector, StopLine
from nuplan.common.maps.maps_datatypes import VectorLayer
from nuplan.common.maps.nuplan_map.baseline_path import NuPlanBaselinePath
from nuplan.common.maps.nuplan_map.stop_line import NuPlanStopLine
from nuplan.common.maps.nuplan_map.utils import get_row_with_value


class NuPlanLaneConnector(LaneConnector):
    """
    NuPlanMap implementation of LaneConnector.
    """

    def __init__(
        self,
        lane_connector_id: str,
        lanes_df: VectorLayer,
        lane_connectors_df: VectorLayer,
        baseline_paths_df: VectorLayer,
        stop_lines_df: VectorLayer,
    ):
        """
        Constructor of NuPlanLaneConnector.
        :param lane_connector_id: unique identifier of the lane connector.
        :param lanes_df: the geopandas GeoDataframe that contains all lanes in the map.
        :param lane_connectors_df: the geopandas GeoDataframe that contains all lane connectors in the map.
        :param baseline_paths_df: the geopandas GeoDataframe that contains all baselines in the map.
        :param stop_lines_df: the geopandas GeoDataframe that contains all stop lines in the map.
        """
        super().__init__(lane_connector_id)
        self._lanes_df = lanes_df
        self._lane_connectors_df = lane_connectors_df
        self._baseline_paths_df = baseline_paths_df
        self._stop_lines_df = stop_lines_df
        self._baseline_path = None
        self._lane_connector = None
        self._polygon = None
        self._stop_lines: Optional[List[StopLine]] = None

    def incoming_edges(self) -> List[GraphEdgeMapObject]:
        """Inherited from superclass."""
        incoming_lane_id = self._get_lane_connector()["exit_lane_fid"]

        return [
            lane.NuPlanLane(
                str(incoming_lane_id),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._stop_lines_df,
            )
        ]

    def outgoing_edges(self) -> List[GraphEdgeMapObject]:
        """Inherited from superclass."""
        outgoing_lane_id = self._get_lane_connector()["entry_lane_fid"]

        return [
            lane.NuPlanLane(
                str(outgoing_lane_id),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._stop_lines_df,
            )
        ]

    def baseline_path(self) -> BaselinePath:
        """Inherited from superclass."""
        if self._baseline_path is None:
            self._baseline_path = NuPlanBaselinePath(
                get_row_with_value(self._baseline_paths_df, "lane_connector_fid", self.id)
            )

        return self._baseline_path

    @property
    def speed_limit_mps(self) -> Optional[float]:
        """Inherited from superclass."""
        speed_limit = self._get_lane_connector()["speed_limit_mps"]
        is_valid = speed_limit == speed_limit and speed_limit is not None
        return float(speed_limit) if is_valid else None

    @property
    def polygon(self) -> Polygon:
        """Inherited from superclass. Note, the polygon is inferred from the baseline."""
        approximate_lane_connector_width = 5.0
        if self._polygon is None:
            linestring = self.baseline_path().linestring
            self._polygon = linestring.buffer(approximate_lane_connector_width / 2, cap_style=CAP_STYLE.flat)

        return self._polygon

    def get_roadblock_id(self) -> str:
        """Inherited from superclass."""
        return str(self._get_lane_connector()["lane_group_connector_fid"])

    def has_traffic_lights(self) -> bool:
        """Inherited from superclass."""
        return bool(self._get_lane_connector()["traffic_light_stop_line_fids"])

    def get_stop_lines(self) -> List[StopLine]:
        """Inherited from superclass."""
        if self._stop_lines is None:
            stop_line_ids = self._get_lane_connector()["traffic_light_stop_line_fids"]
            stop_line_ids = cast(List[str], stop_line_ids.replace(" ", "").split(","))
            self._stop_lines = [NuPlanStopLine(id_, self._stop_lines_df) for id_ in stop_line_ids if id_]

        return self._stop_lines

    def _get_lane_connector(self) -> pd.Series:
        """
        Gets the series from the lane dataframe containing lane's id.
        :return: the respective series from the lanes dataframe.
        """
        if self._lane_connector is None:
            self._lane_connector = get_row_with_value(self._lane_connectors_df, "fid", self.id)

        return self._lane_connector
