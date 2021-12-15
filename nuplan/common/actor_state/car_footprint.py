from enum import IntEnum
from typing import Optional

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.transform_state import translate_longitudinally, translate_longitudinally_se2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters


class CarPointType(IntEnum):
    """ Enum for the point of interest in the car. """
    FRONT_BUMPER = 1,
    REAR_BUMPER = 2,
    REAR_AXLE = 3,
    FRONT_LEFT = 4,
    FRONT_RIGHT = 5,
    REAR_LEFT = 6,
    REAR_RIGHT = 7,
    CENTER = 8


class CarFootprint:
    """ Class that represent the car semantically, with geometry and relevant point of interest. """

    def __init__(self, pose: StateSE2, vehicle_parameters: Optional[VehicleParameters] = None,
                 reference_frame: CarPointType = CarPointType.REAR_AXLE):
        """
        :param pose: The pose of ego in the specified frame
        :param vehicle_parameters: The parameters of ego
        :param reference_frame: Reference frame for the given pose, by default the center of the rear axle
        """

        if vehicle_parameters is None:
            vehicle_parameters = get_pacifica_parameters()

        self._rear_axle_to_center_dist = float(vehicle_parameters.rear_axle_to_center)

        if reference_frame == CarPointType.REAR_AXLE:
            center = translate_longitudinally_se2(pose, self._rear_axle_to_center_dist)
        elif reference_frame == CarPointType.CENTER:
            center = pose
        else:
            raise RuntimeError("Invalid reference frame")

        self._oriented_box = OrientedBox(center, vehicle_parameters.length, vehicle_parameters.width,
                                         vehicle_parameters.height)

        self._points_of_interest = {
            CarPointType.FRONT_BUMPER: translate_longitudinally(self._oriented_box.center,
                                                                self._oriented_box.length / 2.0),
            CarPointType.REAR_BUMPER: translate_longitudinally(self._oriented_box.center,
                                                               - self._oriented_box.length / 2.0),
            CarPointType.REAR_AXLE: translate_longitudinally(self._oriented_box.center,
                                                             - self._rear_axle_to_center_dist),
            CarPointType.FRONT_LEFT: Point2D(*self._oriented_box.geometry.exterior.coords[0]),
            CarPointType.REAR_LEFT: Point2D(*self._oriented_box.geometry.exterior.coords[1]),
            CarPointType.REAR_RIGHT: Point2D(*self._oriented_box.geometry.exterior.coords[2]),
            CarPointType.FRONT_RIGHT: Point2D(*self._oriented_box.geometry.exterior.coords[3]),
            CarPointType.CENTER: Point2D(self._oriented_box.center.x, self._oriented_box.center.y),
        }
        self._rear_axle = translate_longitudinally_se2(self.oriented_box.center, - self._rear_axle_to_center_dist)

    def get_point_of_interest(self, point_of_interest: CarPointType) -> Point2D:
        """
        Getter for the point of interest of ego.
        :param point_of_interest: The query point of the car
        :return: The position of the query point.
        """
        return self._points_of_interest[point_of_interest]

    @property
    def oriented_box(self) -> OrientedBox:
        """ Getter for Ego's OrientedBox
        :return: OrientedBox of Ego
        """
        return self._oriented_box

    @property
    def rear_axle_to_center_dist(self) -> float:
        """ Getter for the distance from the rear axle to the center of mass of Ego.
        :return: Distance from rear axle to COG
        """
        return self._rear_axle_to_center_dist

    @property
    def rear_axle(self) -> StateSE2:
        """ Getter for the pose at the middle of the rear axle
        :return: SE2 Pose of the rear axle.
        """
        return self._rear_axle
