from functools import cached_property
from typing import List, Optional

import pandas as pd
from shapely.geometry import Polygon

import nuplan.common.maps.nuplan_map.intersection as intersection
import nuplan.common.maps.nuplan_map.roadblock as roadblock
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    Intersection,
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
)
from nuplan.common.maps.maps_datatypes import VectorLayer
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.nuplan_map.utils import get_all_rows_with_value, get_row_with_value


class NuPlanRoadBlockConnector(RoadBlockGraphEdgeMapObject):
    """
    NuPlanMap implmentation of Roadblock Connector.
    """

    def __init__(
        self,
        roadblock_connector_id: str,
        lanes_df: VectorLayer,
        lane_connectors_df: VectorLayer,
        baseline_paths_df: VectorLayer,
        boundaries_df: VectorLayer,
        roadblocks_df: VectorLayer,
        roadblock_connectors_df: VectorLayer,
        stop_lines_df: VectorLayer,
        intersections_df: VectorLayer,
        lane_connector_polygon_df: VectorLayer,
        map_data: AbstractMap,
    ):
        """
        Constructor of NuPlanLaneConnector.
        :param roadblock_connector_id: unique identifier of the roadblock connector.
        :param lanes_df: the geopandas GeoDataframe that contains all lanes in the map.
        :param lane_connectors_df: the geopandas GeoDataframe that contains all lane connectors in the map.
        :param baseline_paths_df: the geopandas GeoDataframe that contains all baselines in the map.
        :param boundaries_df: the geopandas GeoDataframe that contains all boundaries in the map.
        :param roadblocks_df: the geopandas GeoDataframe that contains all roadblocks (lane groups) in the map.
        :param roadblock_connectors_df: the geopandas GeoDataframe that contains all roadblock connectors (lane group
            connectors) in the map.
        :param stop_lines_df: the geopandas GeoDataframe that contains all stop lines in the map.
        :param lane_connector_polygon_df: the geopandas GeoDataframe that contains polygons for lane connectors.
        """
        super().__init__(roadblock_connector_id)
        self._lanes_df = lanes_df
        self._lane_connectors_df = lane_connectors_df
        self._baseline_paths_df = baseline_paths_df
        self._boundaries_df = boundaries_df
        self._roadblocks_df = roadblocks_df
        self._roadblock_connectors_df = roadblock_connectors_df
        self._stop_lines_df = stop_lines_df
        self._lane_connector_polygon_df = lane_connector_polygon_df
        self._intersections_df = intersections_df
        self._map_data = map_data

    @cached_property
    def incoming_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        """Inherited from superclass."""
        incoming_roadblock_id = self._roadblock_connector["from_lane_group_fid"]

        return [
            roadblock.NuPlanRoadBlock(
                str(incoming_roadblock_id),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._roadblocks_df,
                self._roadblock_connectors_df,
                self._stop_lines_df,
                self._intersections_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
        ]

    @cached_property
    def outgoing_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        """Inherited from superclass."""
        outgoing_roadblock_id = self._roadblock_connector["to_lane_group_fid"]

        return [
            roadblock.NuPlanRoadBlock(
                str(outgoing_roadblock_id),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._roadblocks_df,
                self._roadblock_connectors_df,
                self._stop_lines_df,
                self._intersections_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
        ]

    @cached_property
    def interior_edges(self) -> List[LaneGraphEdgeMapObject]:
        """Inherited from superclass."""
        lane_connector_ids = get_all_rows_with_value(self._lane_connectors_df, "lane_group_connector_fid", self.id)[
            "fid"
        ]

        return [
            NuPlanLaneConnector(
                str(lane_connector_id),
                self._lanes_df,
                self._lane_connectors_df,
                self._baseline_paths_df,
                self._boundaries_df,
                self._stop_lines_df,
                self._lane_connector_polygon_df,
                self._map_data,
            )
            for lane_connector_id in lane_connector_ids.to_list()
        ]

    @cached_property
    def polygon(self) -> Polygon:
        """Inherited from superclass."""
        return self._roadblock_connector.geometry

    @cached_property
    def children_stop_lines(self) -> List[StopLine]:
        """Inherited from superclass."""
        raise NotImplementedError

    @cached_property
    def parallel_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        """Inherited from superclass."""
        raise NotImplementedError

    @cached_property
    def _roadblock_connector(self) -> pd.Series:
        """
        Gets the series from the roadblock connector dataframe containing roadblock connector's id.
        :return: the respective series from the roadblock connectors dataframe.
        """
        return get_row_with_value(self._roadblock_connectors_df, "fid", self.id)

    @property
    def intersection(self) -> Optional[Intersection]:
        """Inherited from superclass."""
        intersection_id = str(self._roadblock_connector['intersection_fid'])
        return intersection.NuPlanIntersection(intersection_id, self._intersections_df)
