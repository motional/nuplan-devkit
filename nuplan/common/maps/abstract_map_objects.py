from __future__ import annotations

import abc
from typing import List, Optional

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.maps_datatypes import IntersectionType, StopLineType
from shapely.geometry import LineString, Point, Polygon


class AbstractMapObject(abc.ABC):
    """
    Base interface representation of all map objects
    """

    def __init__(self, object_id: str):
        """
        Constructor of the base lane type
        :param object_id: unique identifier of the map object
        """
        self.id = str(object_id)


class PolygonMapObject(AbstractMapObject):
    """
    A class to represent any map object that can be represented as a polygon
    """
    @property
    @abc.abstractmethod
    def polygon(self) -> Polygon:
        """
        Returns the surface of the map object as a Polygon
        :return: The map object as a Polygon
        """
        pass

    def contains_point(self, point: Point2D) -> bool:
        """
        Checks if the specified point is part of the map object polygon
        :return: True if the point is within the polygon
        """
        return bool(self.polygon.contains(Point(point.x, point.y)))


class GraphEdgeMapObject(PolygonMapObject):
    """
    A class to represent any map object that can be an edge as a part of the map graph connectivity
    """

    @abc.abstractmethod
    def incoming_edges(self) -> List[GraphEdgeMapObject]:
        """
        Returns incoming edges connecting to this edge.
        :return: a list of GraphEdgeMapObject
        """
        pass

    @abc.abstractmethod
    def outgoing_edges(self) -> List[GraphEdgeMapObject]:
        """
        Returns outgoing edges from this edge.
        :return: a list of GraphEdgeMapObject
        """
        pass

    @abc.abstractmethod
    def baseline_path(self) -> BaselinePath:
        """
        Getter function for obtaining the baseline path of the lane
        :return: Baseline path of the lane
        """
        pass

    @property
    @abc.abstractmethod
    def speed_limit_mps(self) -> Optional[float]:
        """
        Getter function for obtaining the speed limit of the lane
        :return: [m/s] Speed limit
        """
        pass


class Lane(GraphEdgeMapObject):

    def __init__(self, lane_id: str):
        """
        Constructor of the base lane type
        :param lane_id: unique identifier of the lane
        """
        super().__init__(lane_id)

    @abc.abstractmethod
    def get_roadblock_id(self) -> str:
        """
        Getter function for obtaining the roadblock id containing the lane
        :return: Roadblock ID containing the lane
        """
        pass


class LaneConnector(GraphEdgeMapObject):

    def __init__(self, lane_connector_id: str):
        """
        Constructor of the base lane connector type
        :param lane_connector_id: unique identifier of the lane
        """
        super().__init__(lane_connector_id)


class BaselinePath(AbstractMapObject):

    def __init__(self, path_id: str):
        """
        Constructor of the base lane connector type
        :param path_id: unique identifier of the baseline path
        """
        super().__init__(path_id)

    @property
    @abc.abstractmethod
    def parent(self) -> AbstractMapObject:
        """
        Returns the map object the crosswalk resides in
        :return: The parent map object
        """
        pass

    @property
    @abc.abstractmethod
    def linestring(self) -> LineString:
        """
        Returns the baseline as a Linestring
        :return: The baseline as a Linestring
        """
        pass

    @abc.abstractmethod
    def discrete_path(self) -> List[StateSE2]:
        """
        Gets a discretized representation of the baseline path
        :return: a list of StateSE2
        """
        pass

    @abc.abstractmethod
    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """
        Returns the arc length along the baseline where the given point is the closest
        :param point: [m] x, y coordinates in global frame
        :return: [m] arc length along the baseline
        """
        pass

    @abc.abstractmethod
    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """
        Returns the pose along the baseline where the given point is the closest
        :param point: [m] x, y coordinates in global frame
        :return: nearest pose along the base line as StateSE2
        """
        pass

    @abc.abstractmethod
    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """
        Return curvature at an arc length along the baseline
        :param arc_length: [m] arc length along the baseline. It has to be 0<= arc_length <=length
        :return: [1/m] curvature along a base line
        """
        pass


class StopLine(PolygonMapObject):

    def __init__(self, stop_line_id: str, stop_line_type: StopLineType) -> None:
        """
        Constructor of the base stop line type.
        :param stop_line_id: unique identifier of the stop line.
        :param stop_line_type: stop line sub type. E.g. PED_CROSSING, STOP_SIGN, TRAFFIC_LIGHT, TURN_STOP.
        """
        super().__init__(stop_line_id)
        self.stop_line_type = stop_line_type


class Intersection(PolygonMapObject):

    def __init__(self, intersection_id: str, intersection_type: IntersectionType) -> None:
        """
        Constructor of the base intersection type.
        :param intersection_id: unique identifier of the intersection.
        :param intersection_type: stop line sub type. E.g. DEFAULT, TRAFFIC_LIGHT, STOP_SIGN
        """
        super().__init__(intersection_id)
        self.intersection_type = intersection_type


class Crosswalk(PolygonMapObject):

    def __init__(self, crosswalk_id: str) -> None:
        """
        Constructor of the base stop line type.
        :param crosswalk_id: unique identifier of the lane
        """
        super().__init__(crosswalk_id)
