from functools import cached_property
from typing import List, Optional, Tuple

import pandas as pd
from shapely.geometry import Polygon

import nuplan.common.maps.nuplan_map.lane_connector as lane_connector
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    Lane,
    LaneGraphEdgeMapObject,
    PolylineMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, VectorLayer
from nuplan.common.maps.nuplan_map.polyline_map_object import NuPlanPolylineMapObject
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
        boundaries_df: VectorLayer,
        stop_lines_df: VectorLayer,
        lane_connector_polygon_df: VectorLayer,
        map_data: AbstractMap,
    ):
        """
        Constructor of NuPlanLane.
        :param lane_id: unique identifier of the lane.
        :param lanes_df: the geopandas GeoDataframe that contains all lanes in the map.
        :param lane_connectors_df: the geopandas GeoDataframe that contains all lane connectors in the map.
        :param baseline_paths_df: the geopandas GeoDataframe that contains all baselines in the map.
        :param boundaries_df: the geopandas GeoDataframe that contains all boundaries in the map.
        :param stop_lines_df: the geopandas GeoDataframe that contains all stop lines in the map.
        :param lane_connector_polygon_df: the geopandas GeoDataframe that contains polygons for lane connectors.
        """
        super().__init__(lane_id)
        self._lanes_df = lanes_df
        self._lane_connectors_df = lane_connectors_df
        self._baseline_paths_df = baseline_paths_df
        self._boundaries_df = boundaries_df
        self._stop_lines_df = stop_lines_df
        self._lane_connector_polygon_df = lane_connector_polygon_df
        self._lane = None
        self._map_data = map_data

    @cached_property
    def incoming_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass."""
        lane_connectors_ids = get_all_rows_with_value(self._lane_connectors_df, "entry_lane_fid", self.id)["fid"]

        return [
            lane_connector.NuPlanLaneConnector(
                lane_connector_id,
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._stop_lines_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
            for lane_connector_id in lane_connectors_ids.tolist()
        ]

    @cached_property
    def outgoing_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass."""
        lane_connectors_ids = get_all_rows_with_value(self._lane_connectors_df, "exit_lane_fid", self.id)["fid"]

        return [
            lane_connector.NuPlanLaneConnector(
                lane_connector_id,
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._stop_lines_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
            for lane_connector_id in lane_connectors_ids.to_list()
        ]

    @cached_property
    def parallel_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass"""
        raise NotImplementedError

    @cached_property
    def baseline_path(self) -> PolylineMapObject:
        """Inherited from superclass."""
        return NuPlanPolylineMapObject(get_row_with_value(self._baseline_paths_df, "lane_fid", self.id))

    @cached_property
    def left_boundary(self) -> PolylineMapObject:
        """Inherited from superclass."""
        boundary_fid = self._get_lane()["left_boundary_fid"]
        return NuPlanPolylineMapObject(get_row_with_value(self._boundaries_df, "fid", str(boundary_fid)))

    @cached_property
    def right_boundary(self) -> PolylineMapObject:
        """Inherited from superclass."""
        boundary_fid = self._get_lane()["right_boundary_fid"]
        return NuPlanPolylineMapObject(get_row_with_value(self._boundaries_df, "fid", str(boundary_fid)))

    def get_roadblock_id(self) -> str:
        """Inherited from superclass."""
        return str(self._get_lane()["lane_group_fid"])

    @cached_property
    def parent(self) -> RoadBlockGraphEdgeMapObject:
        """Inherited from superclass"""
        return self._map_data.get_map_object(self.get_roadblock_id(), SemanticMapLayer.ROADBLOCK)

    @cached_property
    def speed_limit_mps(self) -> Optional[float]:
        """Inherited from superclass."""
        speed_limit = self._get_lane()["speed_limit_mps"]
        is_valid = speed_limit == speed_limit and speed_limit is not None
        return float(speed_limit) if is_valid else None

    @cached_property
    def polygon(self) -> Polygon:
        """Inherited from superclass."""
        return self._get_lane().geometry

    def is_left_of(self, other: Lane) -> bool:
        """Inherited from superclass."""
        assert self.is_same_roadblock(other), "Lanes must be in the same roadblock"

        other_lane = get_row_with_value(self._lanes_df, "fid", other.id)
        other_index = int(other_lane["lane_index"])
        self_index = int(self._get_lane()["lane_index"])
        return self_index < other_index

    def is_right_of(self, other: Lane) -> bool:
        """Inherited from superclass."""
        assert self.is_same_roadblock(other), "Lanes must be in the same roadblock"

        other_lane = get_row_with_value(self._lanes_df, "fid", other.id)
        other_index = int(other_lane["lane_index"])
        self_index = int(self._get_lane()["lane_index"])
        return self_index > other_index

    @cached_property
    def adjacent_edges(self) -> Tuple[Optional[LaneGraphEdgeMapObject], Optional[LaneGraphEdgeMapObject]]:
        """Inherited from superclass."""
        lane_group_fid = self._get_lane()["lane_group_fid"]
        all_lanes = get_all_rows_with_value(self._lanes_df, "lane_group_fid", lane_group_fid)

        lane_index = self._get_lane()["lane_index"]
        # According to the map attributes, lanes are numbered left to right with smaller indices being on the left and larger indices being on the right
        left_lane_id = all_lanes[all_lanes["lane_index"] == int(lane_index) - 1]["fid"]
        right_lane_id = all_lanes[all_lanes["lane_index"] == int(lane_index) + 1]["fid"]

        left_lane = (
            NuPlanLane(
                left_lane_id.item(),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._stop_lines_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
            if not left_lane_id.empty
            else None
        )
        right_lane = (
            NuPlanLane(
                right_lane_id.item(),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._stop_lines_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
            if not right_lane_id.empty
            else None
        )

        return left_lane, right_lane

    def get_width_left_right(
        self, point: Point2D, include_outside: bool = False
    ) -> Tuple[Optional[float], Optional[float]]:
        """Inherited from superclass."""
        raise NotImplementedError

    def oriented_distance(self, point: Point2D) -> float:
        """Inherited from superclass"""
        raise NotImplementedError

    def _get_lane(self) -> pd.Series:
        """
        Gets the series from the lane dataframe containing lane's id.
        :return: the respective series from the lanes dataframe.
        """
        if self._lane is None:
            self._lane = get_row_with_value(self._lanes_df, "fid", self.id)

        return self._lane

    @cached_property
    def index(self) -> int:
        """Inherited from superclass"""
        return int(self._get_lane()["lane_index"])
