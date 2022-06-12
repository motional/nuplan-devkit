from dataclasses import dataclass
from typing import Optional


@dataclass
class BoxParameters:
    """Class describing a planar box dimensions"""

    width: float  # [m] Box width
    length: float  # [m] Box length

    @property
    def half_width(self) -> float:
        """
        Getter for half width of the box
        :return: half the width
        """
        return self.width / 2.0

    @property
    def half_length(self) -> float:
        """
        Getter for half length of the box
        :return: half the length
        """
        return self.length / 2.0


class VehicleParameters(BoxParameters):
    """
    Class holding parameters of a vehicle
    """

    def __init__(
        self,
        width: float,
        front_length: float,
        rear_length: float,
        cog_position_from_rear_axle: float,
        wheel_base: float,
        height: Optional[float] = None,
    ):
        """
        :param width: [m] width of box around vehicle
        :param front_length: [m] distance between rear axle and front bumper
        :param rear_length: [m] distance between rear axle and rear bumper
        :param cog_position_from_rear_axle: [m] distance between rear axle and center of gravity (cog)
        :param wheel_base: [m] wheel base of the vehicle
        :param height: [m] height of box around vehicle
        """
        self.width = width
        self.front_length = front_length
        self.rear_length = rear_length
        self.wheel_base = wheel_base
        self.length = front_length + rear_length
        self.cog_position_from_rear_axle = cog_position_from_rear_axle
        self.height = height

        # [m] (dist. from rear axle to front bumper)
        self.front_length = front_length
        # [m] (dist. from rear axel to rear bumper)
        self.rear_length = rear_length
        # [m] position of COG wrt. to rear axle
        self.cog_position_from_rear_axle = cog_position_from_rear_axle

    @property
    def rear_axle_to_center(self) -> float:
        """
        :return: [m] distance between rear axle and center of vehicle
        """
        return self.half_length - self.rear_length

    @property
    def length_cog_to_front_axle(self) -> float:
        """
        :return: [m] distance between cog and front axle
        """
        return self.wheel_base - self.cog_position_from_rear_axle


def get_pacifica_parameters() -> VehicleParameters:
    """
    :return VehicleParameters containing parameters of Pacifica Vehicle
    """
    return VehicleParameters(
        width=1.1485 * 2.0,
        front_length=4.049,
        rear_length=1.127,
        wheel_base=3.089,
        cog_position_from_rear_axle=1.67,
        height=1.777,
    )
