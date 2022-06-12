from typing import List, Optional

import pandas as pd
from shapely.geometry import Polygon

import nuplan.common.maps.nuplan_map.lane_connector as lane_connector
from nuplan.common.maps.abstract_map_objects import BaselinePath, GraphEdgeMapObject, Lane
from nuplan.common.maps.maps_datatypes import VectorLayer
from nuplan.common.maps.nuplan_map.baseline_path import NuPlanBaselinePath
from nuplan.common.maps.nuplan_map.utils import get_all_rows_with_value, get_row_with_value


class NuPlanLane(Lane):
    """
    NuPlanMap implementation of Lane.
    """

    def __init__(
        self,
        lane_id: str,
        lanes_df: VectorLayer,
        lane_connectors_df: VectorLayer,
        baseline_paths_df: VectorLayer,
        stop_lines_df: VectorLayer,
    ):
        """
        Constructor of NuPlanLane.
        :param lane_id: unique identifier of the lane.
        :param lanes_df: the geopandas GeoDataframe that contains all lanes in the map.
        :param lane_connectors_df: the geopandas GeoDataframe that contains all lane connectors in the map.
        :param baseline_paths_df: the geopandas GeoDataframe that contains all baselines in the map.
        :param stop_lines_df: the geopandas GeoDataframe that contains all stop lines in the map.
        """
        super().__init__(lane_id)
        self._lanes_df = lanes_df
        self._lane_connectors_df = lane_connectors_df
        self._baseline_paths_df = baseline_paths_df
        self._stop_lines_df = stop_lines_df
        self._baseline_path = None
        self._lane = None

    def incoming_edges(self) -> List[GraphEdgeMapObject]:
        """Inherited from superclass."""
        lane_connectors_ids = get_all_rows_with_value(self._lane_connectors_df, "entry_lane_fid", self.id)["fid"]

        return [
            lane_connector.NuPlanLaneConnector(
                lane_connector_id,
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._stop_lines_df,
            )
            for lane_connector_id in lane_connectors_ids.tolist()
        ]

    def outgoing_edges(self) -> List[GraphEdgeMapObject]:
        """Inherited from superclass."""
        lane_connectors_ids = get_all_rows_with_value(self._lane_connectors_df, "exit_lane_fid", self.id)["fid"]

        return [
            lane_connector.NuPlanLaneConnector(
                lane_connector_id,
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._stop_lines_df,
            )
            for lane_connector_id in lane_connectors_ids.to_list()
        ]

    def baseline_path(self) -> BaselinePath:
        """Inherited from superclass."""
        if self._baseline_path is None:
            self._baseline_path = NuPlanBaselinePath(get_row_with_value(self._baseline_paths_df, "lane_fid", self.id))

        return self._baseline_path

    def get_roadblock_id(self) -> str:
        """Inherited from superclass."""
        return str(self._get_lane()["lane_group_fid"])

    @property
    def speed_limit_mps(self) -> Optional[float]:
        """Inherited from superclass."""
        speed_limit = self._get_lane()["speed_limit_mps"]
        is_valid = speed_limit == speed_limit and speed_limit is not None
        return float(speed_limit) if is_valid else None

    @property
    def polygon(self) -> Polygon:
        """Inherited from superclass."""
        return self._get_lane().geometry

    def _get_lane(self) -> pd.Series:
        """
        Gets the series from the lane dataframe containing lane's id.
        :return: the respective series from the lanes dataframe.
        """
        if self._lane is None:
            self._lane = get_row_with_value(self._lanes_df, "fid", self.id)

        return self._lane
