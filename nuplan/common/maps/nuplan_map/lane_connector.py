from functools import cached_property
from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

import nuplan.common.maps.nuplan_map.lane as lane
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    LaneConnector,
    LaneGraphEdgeMapObject,
    PolylineMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
)
from nuplan.common.maps.maps_datatypes import LaneConnectorType, SemanticMapLayer, VectorLayer
from nuplan.common.maps.nuplan_map.polyline_map_object import NuPlanPolylineMapObject
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
        boundaries_df: VectorLayer,
        stop_lines_df: VectorLayer,
        lane_connector_polygon_df: VectorLayer,
        map_data: AbstractMap,
    ):
        """
        Constructor of NuPlanLaneConnector.
        :param lane_connector_id: unique identifier of the lane connector.
        :param lanes_df: the geopandas GeoDataframe that contains all lanes in the map.
        :param lane_connectors_df: the geopandas GeoDataframe that contains all lane connectors in the map.
        :param baseline_paths_df: the geopandas GeoDataframe that contains all baselines in the map.
        :param boundaries_df: the geopandas GeoDataframe that contains all boundaries in the map.
        :param stop_lines_df: the geopandas GeoDataframe that contains all stop lines in the map.
        :param lane_connector_polygon_df: the geopandas GeoDataframe that contains polygons for lane connectors.
        """
        super().__init__(lane_connector_id)
        self._lanes_df = lanes_df
        self._lane_connectors_df = lane_connectors_df
        self._baseline_paths_df = baseline_paths_df
        self._boundaries_df = boundaries_df
        self._stop_lines_df = stop_lines_df
        self._lane_connector_polygon_df = lane_connector_polygon_df
        self._lane_connector = None
        self._map_data = map_data

    @cached_property
    def incoming_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass."""
        incoming_lane_id = self._get_lane_connector()["exit_lane_fid"]

        return [
            lane.NuPlanLane(
                str(incoming_lane_id),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._stop_lines_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
        ]

    @cached_property
    def outgoing_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass."""
        outgoing_lane_id = self._get_lane_connector()["entry_lane_fid"]

        return [
            lane.NuPlanLane(
                str(outgoing_lane_id),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._stop_lines_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
        ]

    @cached_property
    def parallel_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass"""
        raise NotImplementedError

    @cached_property
    def baseline_path(self) -> PolylineMapObject:
        """Inherited from superclass."""
        return NuPlanPolylineMapObject(get_row_with_value(self._baseline_paths_df, "lane_connector_fid", self.id))

    @cached_property
    def left_boundary(self) -> PolylineMapObject:
        """Inherited from superclass."""
        boundary_fid = get_row_with_value(self._lane_connector_polygon_df, "lane_connector_fid", self.id)[
            "left_boundary_fid"
        ]
        return NuPlanPolylineMapObject(get_row_with_value(self._boundaries_df, "fid", str(boundary_fid)))

    @cached_property
    def right_boundary(self) -> PolylineMapObject:
        """Inherited from superclass."""
        boundary_fid = get_row_with_value(self._lane_connector_polygon_df, "lane_connector_fid", self.id)[
            "right_boundary_fid"
        ]
        return NuPlanPolylineMapObject(get_row_with_value(self._boundaries_df, "fid", str(boundary_fid)))

    @cached_property
    def speed_limit_mps(self) -> Optional[float]:
        """Inherited from superclass."""
        speed_limit = self._get_lane_connector()["speed_limit_mps"]
        is_valid = speed_limit == speed_limit and speed_limit is not None
        return float(speed_limit) if is_valid else None

    @cached_property
    def polygon(self) -> Polygon:
        """Inherited from superclass. Note, the polygon is inferred from the baseline."""
        lane_connector_polygon_row = get_row_with_value(self._lane_connector_polygon_df, "lane_connector_fid", self.id)
        return lane_connector_polygon_row.geometry

    def is_left_of(self, other: LaneConnector) -> bool:
        """Inherited from superclass."""
        # Due to lack of lane connector adjacency information, this always returns false
        return False

    def is_right_of(self, other: LaneConnector) -> bool:
        """Inherited from superclass."""
        # Due to lack of lane connector adjacency information, this always returns false
        return False

    def get_roadblock_id(self) -> str:
        """Inherited from superclass."""
        return str(self._get_lane_connector()["lane_group_connector_fid"])

    @cached_property
    def parent(self) -> RoadBlockGraphEdgeMapObject:
        """Inherited from superclass"""
        return self._map_data.get_map_object(self.get_roadblock_id(), SemanticMapLayer.ROADBLOCK_CONNECTOR)

    def has_traffic_lights(self) -> bool:
        """Inherited from superclass."""
        return bool(self._get_lane_connector()["traffic_light_stop_line_fids"])

    @cached_property
    def stop_lines(self) -> List[StopLine]:
        """Inherited from superclass."""
        stop_line_ids = self._get_lane_connector()["traffic_light_stop_line_fids"]
        stop_line_ids = cast(List[str], stop_line_ids.replace(" ", "").split(","))

        candidate_stop_lines = [NuPlanStopLine(id_, self._stop_lines_df) for id_ in stop_line_ids if id_]

        # This lane connector has no stop lines associated
        if not candidate_stop_lines:
            return []

        stop_lines = [
            stop_line
            for stop_line in candidate_stop_lines
            if stop_line.polygon.intersects(self.baseline_path.linestring)
        ]

        # If intersection check is successful then return stop lines.
        if stop_lines:
            return stop_lines

        # Stop line is not intersecting the lane connector's baseline. Perform a distance check instead.
        def distance_to_stop_line(stop_line: StopLine) -> float:
            """
            Calculates the distance between the first point of the lane connector's baseline path
            :param stop_line: The stop line to calculate the distance to.
            :return: [m] The distance between first point points of the lane connector to the stop_line polygon.
            """
            start = Point(self.baseline_path.linestring.coords[0])
            return float(start.distance(stop_line.polygon))

        distances = [distance_to_stop_line(stop_line) for stop_line in candidate_stop_lines]

        return [candidate_stop_lines[np.argmin(distances)]]

    def turn_type(self) -> LaneConnectorType:
        """Inherited from superclass"""
        raise NotImplementedError

    def get_width_left_right(
        self, point: Point2D, include_outside: bool = False
    ) -> Tuple[Optional[float], Optional[float]]:
        """Inherited from superclass."""
        raise NotImplementedError

    def oriented_distance(self, point: Point2D) -> float:
        """Inherited from superclass"""
        raise NotImplementedError

    def _get_lane_connector(self) -> pd.Series:
        """
        Gets the series from the lane dataframe containing lane's id.
        :return: the respective series from the lanes dataframe.
        """
        if self._lane_connector is None:
            self._lane_connector = get_row_with_value(self._lane_connectors_df, "fid", self.id)

        return self._lane_connector
