from __future__ import annotations

import abc
from typing import List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.maps_datatypes import IntersectionType, LaneConnectorType, StopLineType


class AbstractMapObject(abc.ABC):
    """
    Base interface representation of all map objects.
    """

    def __init__(self, object_id: str):
        """
        Constructor of the base lane type.
        :param object_id: unique identifier of the map object.
        """
        self.id = str(object_id)


class PolygonMapObject(AbstractMapObject):
    """
    A class to represent any map object that can be represented as a polygon.
    """

    @property
    @abc.abstractmethod
    def polygon(self) -> Polygon:
        """
        Returns the surface of the map object as a Polygon.
        :return: The map object as a Polygon.
        """
        pass

    def contains_point(self, point: Point2D) -> bool:
        """
        Checks if the specified point is part of the map object polygon.
        :return: True if the point is within the polygon.
        """
        return bool(self.polygon.contains(Point(point.x, point.y)))


class GraphEdgeMapObject(PolygonMapObject):
    """
    A class to represent any map object that can be an edge as a part of the map graph connectivity.
    """

    @property
    @abc.abstractmethod
    def incoming_edges(self) -> List[GraphEdgeMapObject]:
        """
        Returns incoming edges connecting to this edge.
        :return: a list of GraphEdgeMapObject.
        """
        pass

    @property
    @abc.abstractmethod
    def outgoing_edges(self) -> List[GraphEdgeMapObject]:
        """
        Returns outgoing edges from this edge.
        :return: a list of GraphEdgeMapObject.
        """
        pass

    @property
    @abc.abstractmethod
    def parallel_edges(self) -> List[GraphEdgeMapObject]:
        """
        Returns edges parallel to this edge including itself.
        :return: a list of GraphEdgeMapObjects.
        """
        pass


class LaneGraphEdgeMapObject(GraphEdgeMapObject):
    """
    A class to represent a map object that can be an edge as a part of the map graph connectivity and contains a
    BaselinePath within it.
    """

    @property
    @abc.abstractmethod
    def incoming_edges(self) -> List[LaneGraphEdgeMapObject]:  # type: ignore
        """
        Returns incoming edges connecting to this edge.
        :return: a list of LaneGraphEdgeMapObject.
        """
        pass

    @property
    @abc.abstractmethod
    def outgoing_edges(self) -> List[LaneGraphEdgeMapObject]:  # type: ignore
        """
        Returns outgoing edges from this edge.
        :return: a list of LaneGraphEdgeMapObject.
        """
        pass

    @property
    @abc.abstractmethod
    def baseline_path(self) -> PolylineMapObject:
        """
        Getter function for obtaining the baseline path of the lane.
        :return: Baseline path of the lane.
        """
        pass

    @property
    @abc.abstractmethod
    def left_boundary(self) -> PolylineMapObject:
        """
        Getter function for obtaining the left boundary of the lane.
        :return: Left boundary of the lane.
        """
        pass

    @property
    @abc.abstractmethod
    def right_boundary(self) -> PolylineMapObject:
        """
        Getter function for obtaining the right boundary of the lane.
        :return: Right boundary of the lane.
        """
        pass

    @property
    @abc.abstractmethod
    def speed_limit_mps(self) -> Optional[float]:
        """
        Getter function for obtaining the speed limit of the lane.
        :return: [m/s] Speed limit.
        """
        pass

    @abc.abstractmethod
    def get_roadblock_id(self) -> str:
        """
        Getter function for obtaining the roadblock id containing the lane.
        :return: Roadblock ID containing the lane.
        """
        pass

    @abc.abstractmethod
    def parent(self) -> RoadBlockGraphEdgeMapObject:
        """
        Getter function for obtaining the parent RoadBlockGraphEdgeMapObject containing the LaneGraphEdgeMapObject.
        :return: RoadblockBlockGraphEdgeMapObject containing the LaneGraphEdgeMapObject.
        """
        pass

    @abc.abstractmethod
    def has_traffic_lights(self) -> bool:
        """
        Returns whether this graph edge is controlled by traffic lights.
        :return: True if the edge is controlled by traffic lights. False otherwise.
        """
        pass

    @property
    @abc.abstractmethod
    def stop_lines(self) -> List[StopLine]:
        """
        Returns a list of stop lines associated with this lane connector.
        :return: A list of stop lines associated with this lane connector.
        """
        pass

    def is_same_roadblock(self, other: Lane) -> bool:
        """
        :param other: Lane to check if it is in the same roadblock as self.
        :return: True if lanes are in the same roadblock.
        """
        return self.get_roadblock_id() == other.get_roadblock_id()

    def is_adjacent_to(self, other: Lane) -> bool:
        """
        :param other: Lane to check if it is adjacent to self.
        :return: True if self and other are in the same roadblock and adjacent.
        """
        # Due to lack of lane connector adjacency information, lane connector boundaries do not align so this is always False
        return self.is_same_roadblock(other) and (
            self.right_boundary.id == other.left_boundary.id or self.left_boundary.id == other.right_boundary.id
        )

    @abc.abstractmethod
    def is_left_of(self, other: Lane) -> bool:
        """
        :param other: Lane to check if self is left of.
        :return: True if self and other are in the same RoadBlock and self is anywhere to the left of other.
        :raise AssertionError: if lanes are not in the same RoadBlock.
        """
        pass

    @abc.abstractmethod
    def is_right_of(self, other: Lane) -> bool:
        """
        :param other: Lane to check if self is right of.
        :return: True if self and other are in the same RoadBlock and self is anywhere to the right of other.
        :raise AssertionError: if lanes are not in the same RoadBlock.
        """
        pass

    @property
    @abc.abstractmethod
    def adjacent_edges(self) -> Tuple[Optional[LaneGraphEdgeMapObject], Optional[LaneGraphEdgeMapObject]]:
        """
        Gets adjacent LaneGraphEdgeMapObjects.
        :return: Tuple of adjacent LaneGraphEdgeMapObjects where first element is the left lane and the second element is the right lane.
        """
        pass

    @abc.abstractmethod
    def get_width_left_right(self, point: Point2D, include_outside: bool = False) -> Tuple[float, float]:
        """
        Gets distance to left and right sides of the lane from point.
        :param point: Point in global frame.
        :param include_outside: Allow point to be outside of lane.
        :return: The distance to left and right sides of the lane. If the query is invalid, inf is returned.
            If point is outside the LaneGraphEdgeMapObject and cannot be projected onto the LaneGraphEdgeMapObject and
            include_outside is True then the distance to the edge on the nearest end is returned.
        """
        pass

    @abc.abstractmethod
    def oriented_distance(self, point: Point2D) -> float:
        """
        Calculate the distance between the edge and a point with an oriented distance.
        :param point: Point global frame.
        :return: The distance between the edge and a point with an oriented distance. If the point is outside of the interval of
            the LaneGraphEdgeMapObject then the L1 distance to the nearest end is returned. The distance is positive if the
            point is on the left side of the line, while it is negative if the point is on the right side of the line.
        """
        pass


class Lane(LaneGraphEdgeMapObject):
    """
    Class representing lanes.
    """

    def __init__(self, lane_id: str):
        """
        Constructor of the base lane type.
        :param lane_id: unique identifier of the lane.
        """
        super().__init__(lane_id)

    def has_traffic_lights(self) -> bool:
        """Inherited from superclass."""
        return False

    @property
    def stop_lines(self) -> List[StopLine]:
        """Inherited from superclass."""
        return []

    @abc.abstractmethod
    def index(self) -> int:
        """
        Gets the 1-index position of the lane within the parent roadblock.
        :return: The index of lane.
        """
        pass


class LaneConnector(LaneGraphEdgeMapObject):
    """
    Class representing lane connectors.
    """

    def __init__(self, lane_connector_id: str):
        """
        Constructor of the base lane connector type.
        :param lane_connector_id: unique identifier of the lane.
        """
        super().__init__(lane_connector_id)

    @property
    def adjacent_edges(self) -> Tuple[Optional[LaneGraphEdgeMapObject], Optional[LaneGraphEdgeMapObject]]:
        """Inherited from superclass."""
        # Returns None for both elements since we currently don't have a way of telling if two lane connectors are adjacent
        return None, None

    @property
    @abc.abstractmethod
    def turn_type(self) -> LaneConnectorType:
        """
        Gets the turn type of the lane connector
        :return: LaneConnectorType of lane connector if lane connector has a type else None
        """
        pass


class PolylineMapObject(AbstractMapObject):
    """
    A class to represent any map object that can be represented as a polyline.
    """

    def __init__(self, path_id: str):
        """
        Constructor of the PolylineMapObject type.
        :param path_id: unique identifier of the polyline.
        """
        super().__init__(path_id)

    @property
    @abc.abstractmethod
    def linestring(self) -> LineString:
        """
        Returns the polyline as a Linestring.
        :return: The polyline as a Linestring.
        """
        pass

    @property
    @abc.abstractmethod
    def length(self) -> float:
        """
        Returns the length of the polyline [m].
        :return: the length of the polyline.
        """
        pass

    @property
    @abc.abstractmethod
    def discrete_path(self) -> List[StateSE2]:
        """
        Gets a discretized representation of the polyline.
        :return: a list of StateSE2.
        """
        pass

    @abc.abstractmethod
    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """
        Returns the arc length along the polyline where the given point is the closest.
        :param point: [m] x, y coordinates in global frame.
        :return: [m] arc length along the polyline.
        """
        pass

    @abc.abstractmethod
    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """
        Returns the pose along the polyline where the given point is the closest.
        :param point: [m] x, y coordinates in global frame.
        :return: nearest pose along the polyline as StateSE2.
        """
        pass

    @abc.abstractmethod
    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """
        Return curvature at an arc length along the polyline.
        :param arc_length: [m] arc length along the polyline. It has to be 0<= arc_length <=length.
        :return: [1/m] curvature along a polyline.
        """
        pass

    def get_nearest_curvature_from_position(self, point: Point2D) -> float:
        """
        Returns the curvature along the polyline where the given point is the closest.
        :param point: [m] x, y coordinates in global frame.
        :return: [1/m] curvature along a polyline.
        """
        return self.get_curvature_at_arc_length(self.get_nearest_arc_length_from_position(point))


class RoadBlockGraphEdgeMapObject(GraphEdgeMapObject):
    """
    A class to represent a map object that can be an edge as a part of the map graph connectivity and contains
    instances of LaneGraphEdgeMapObject within it.
    """

    @property
    @abc.abstractmethod
    def incoming_edges(self) -> List[RoadBlockGraphEdgeMapObject]:  # type: ignore
        """
        Returns incoming edges connecting to this edge.
        :return: a list of RoadBlockGraphEdgeMapObject.
        """
        pass

    @property
    @abc.abstractmethod
    def outgoing_edges(self) -> List[RoadBlockGraphEdgeMapObject]:  # type: ignore
        """
        Returns outgoing edges from this edge.
        :return: a list of RoadBlockGraphEdgeMapObject.
        """
        pass

    @property
    @abc.abstractmethod
    def interior_edges(self) -> List[LaneGraphEdgeMapObject]:
        """
        Returns LaneGraphEdgeMapObject edges that comprise this container edge.
        :return: a list of LaneGraphEdgeMapObject.
        """
        pass

    @property
    @abc.abstractmethod
    def children_stop_lines(self) -> List[StopLine]:
        """
        Returns StopLines within this RoadBlockGraphEdgeMapObject.
        :return: a list of StopLines.
        """
        pass

    def intersection(self) -> Optional[Intersection]:
        """
        :return: The intersection parent to the RoadBlockGraphEdgeMapObject, if available.
        """
        pass


class StopLine(PolygonMapObject):
    """
    Class representing stop lines.
    """

    def __init__(self, stop_line_id: str, stop_line_type: StopLineType) -> None:
        """
        Constructor of the base stop line type.
        :param stop_line_id: unique identifier of the stop line.
        :param stop_line_type: stop line sub type. E.g. PED_CROSSING, STOP_SIGN, TRAFFIC_LIGHT, TURN_STOP.
        """
        super().__init__(stop_line_id)
        self.stop_line_type = stop_line_type

    @property
    @abc.abstractmethod
    def intersection_from(self) -> Intersection:
        """
        Gets the related intersection.
        :return: Intersection related to StopLine.
        """
        pass

    @property
    @abc.abstractmethod
    def layer_type(self) -> StopLineType:
        """
        Gets StopLineType for Stopline subtype.
        :return: StopLineType subtype.
        """
        pass

    @property
    @abc.abstractmethod
    def parent(self) -> RoadBlockGraphEdgeMapObject:
        """
        Getter function for obtaining the parent RoadBlockGraphEdgeMapObject containing the StopLine.
        :return: RoadblockBlockGraphEdgeMapObject containing the StopLine.
        """
        pass


class Intersection(PolygonMapObject):
    """
    Class representing intersections.
    """

    def __init__(self, intersection_id: str, intersection_type: IntersectionType) -> None:
        """
        Constructor of the base intersection type.
        :param intersection_id: unique identifier of the intersection.
        :param intersection_type: stop line sub type. E.g. DEFAULT, TRAFFIC_LIGHT, STOP_SIGN.
        """
        super().__init__(intersection_id)
        self.intersection_type = intersection_type

    @property
    @abc.abstractmethod
    def interior_edges(self) -> List[RoadBlockGraphEdgeMapObject]:
        """
        Returns RoadBlockGraphEdgeMapObjects contained within the intersection.
        :return: a list of RoadBlockGraphEdgeMapObject.
        """
        pass

    @property
    @abc.abstractmethod
    def incoming_edges(self) -> List[Lane]:
        """
        Returns incoming Lanes connecting to this intersection.
        :return: a list of Lane.
        """
        pass

    @property
    @abc.abstractmethod
    def is_signaled(self) -> bool:
        """
        Returns if intersection is signaled.
        :return: True if intersection is a traffic light or one of the interior edges has a traffic light is signaled else False.
        """
        pass
