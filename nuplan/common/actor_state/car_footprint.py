from __future__ import annotations

from functools import cached_property

from nuplan.common.actor_state.oriented_box import OrientedBox, OrientedBoxPointType
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.transform import translate_longitudinally


class CarFootprint(OrientedBox):
    """Class that represent the car semantically, with geometry and relevant point of interest."""

    def __init__(self, center: StateSE2, vehicle_parameters: VehicleParameters):
        """
        :param center: The pose of ego in the specified frame
        :param vehicle_parameters: The parameters of ego
        """
        super().__init__(
            center=center,
            width=vehicle_parameters.width,
            length=vehicle_parameters.length,
            height=vehicle_parameters.height,
        )
        self._vehicle_parameters = vehicle_parameters

    @property
    def vehicle_parameters(self) -> VehicleParameters:
        """
        :return: vehicle parameters corresponding to the footprint
        """
        return self._vehicle_parameters

    def get_point_of_interest(self, point_of_interest: OrientedBoxPointType) -> Point2D:
        """
        Getter for the point of interest of ego.
        :param point_of_interest: The query point of the car
        :return: The position of the query point.
        """
        return self.corner(point_of_interest)

    @property
    def oriented_box(self) -> OrientedBox:
        """
        Getter for Ego's OrientedBox
        :return: OrientedBox of Ego
        """
        return self

    @property
    def rear_axle_to_center_dist(self) -> float:
        """
        Getter for the distance from the rear axle to the center of mass of Ego.
        :return: Distance from rear axle to COG
        """
        return float(self._vehicle_parameters.rear_axle_to_center)

    @cached_property
    def rear_axle(self) -> StateSE2:
        """
        Getter for the pose at the middle of the rear axle
        :return: SE2 Pose of the rear axle.
        """
        return translate_longitudinally(self.oriented_box.center, -self.rear_axle_to_center_dist)

    @classmethod
    def build_from_rear_axle(cls, rear_axle_pose: StateSE2, vehicle_parameters: VehicleParameters) -> CarFootprint:
        """
        Construct Car Footprint from rear axle position
        :param rear_axle_pose: SE2 position of rear axle
        :param vehicle_parameters: parameters of vehicle
        :return: CarFootprint
        """
        center = translate_longitudinally(rear_axle_pose, vehicle_parameters.rear_axle_to_center)
        return cls(center=center, vehicle_parameters=vehicle_parameters)

    @classmethod
    def build_from_cog(cls, cog_pose: StateSE2, vehicle_parameters: VehicleParameters) -> CarFootprint:
        """
        Construct Car Footprint from COG position
        :param cog_pose: SE2 position of COG
        :param vehicle_parameters: parameters of vehicle
        :return: CarFootprint
        """
        cog_to_center = vehicle_parameters.rear_axle_to_center - vehicle_parameters.cog_position_from_rear_axle
        center = translate_longitudinally(cog_pose, cog_to_center)
        return cls(center=center, vehicle_parameters=vehicle_parameters)

    @classmethod
    def build_from_center(cls, center: StateSE2, vehicle_parameters: VehicleParameters) -> CarFootprint:
        """
        Construct Car Footprint from geometric center of vehicle
        :param center: SE2 position of geometric center of vehicle
        :param vehicle_parameters: parameters of vehicle
        :return: CarFootprint
        """
        return cls(center=center, vehicle_parameters=vehicle_parameters)
