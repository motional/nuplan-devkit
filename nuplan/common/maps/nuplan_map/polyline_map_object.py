import math
from functools import cached_property
from typing import List, cast

from pandas.core.series import Series
from shapely.geometry import LineString, Point

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map_objects import PolylineMapObject
from nuplan.common.maps.nuplan_map.utils import estimate_curvature_along_path, extract_discrete_polyline


def _get_heading(pt1: Point, pt2: Point) -> float:
    """
    Computes the angle two points makes to the x-axis.
    :param pt1: origin point.
    :param pt2: end point.
    :return: [rad] resulting angle.
    """
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.atan2(y_diff, x_diff)


class NuPlanPolylineMapObject(PolylineMapObject):
    """
    NuPlanMap implementation of Polyline Map Object.
    """

    def __init__(
        self,
        polyline: Series,
        distance_for_curvature_estimation: float = 2.0,
        distance_for_heading_estimation: float = 0.5,
    ):
        """
        Constructor of polyline map layer.
        :param polyline: a pandas series representing the polyline.
        :param distance_for_curvature_estimation: [m] distance of the split between 3-points curvature estimation.
        :param distance_for_heading_estimation: [m] distance between two points on the polyline to calculate
                                                    the relative heading.
        """
        super().__init__(polyline["fid"])
        self._polyline: LineString = polyline.geometry
        assert self._polyline.length > 0.0, "The length of the polyline has to be greater than 0!"

        self._distance_for_curvature_estimation = distance_for_curvature_estimation
        self._distance_for_heading_estimation = distance_for_heading_estimation

    @property
    def linestring(self) -> LineString:
        """Inherited from superclass."""
        return self._polyline

    @property
    def length(self) -> float:
        """Inherited from superclass."""
        return float(self._polyline.length)

    @cached_property
    def discrete_path(self) -> List[StateSE2]:
        """Inherited from superclass."""
        return cast(List[StateSE2], extract_discrete_polyline(self._polyline))

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """Inherited from superclass."""
        return self._polyline.project(Point(point.x, point.y))  # type: ignore

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """Inherited from superclass."""
        arc_length = self.get_nearest_arc_length_from_position(point)
        state1 = self._polyline.interpolate(arc_length)
        state2 = self._polyline.interpolate(arc_length + self._distance_for_heading_estimation)

        if state1 == state2:
            # Handle the case where the queried position (state1) is at the end of the baseline path
            state2 = self._polyline.interpolate(arc_length - self._distance_for_heading_estimation)
            heading = _get_heading(state2, state1)
        else:
            heading = _get_heading(state1, state2)

        return StateSE2(state1.x, state1.y, heading)

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """Inherited from superclass."""
        curvature = estimate_curvature_along_path(self._polyline, arc_length, self._distance_for_curvature_estimation)

        return float(curvature)
