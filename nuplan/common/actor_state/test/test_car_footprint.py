import unittest

import numpy as np
from nuplan.common.actor_state.car_footprint import CarFootprint, CarPointType
from nuplan.common.actor_state.test.test_utils import get_sample_pose
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


class TestCarFootprint(unittest.TestCase):
    def setUp(self) -> None:
        self.center_position_from_rear_axle = get_pacifica_parameters().rear_axle_to_center

    def test_car_footprint_creation(self) -> None:
        """ Checks that the car footprint is created correctly, in particular the point of interest. """
        car_footprint = CarFootprint(get_sample_pose(), get_pacifica_parameters())

        self.assertAlmostEqual(car_footprint.rear_axle_to_center_dist, self.center_position_from_rear_axle)

        # Check that the point of interest are created correctly
        expected_values = {
            "fb": (1.0, 6.049),
            "rb": (1.0, 0.873),
            "ra": (1.0, 2.0),
            "fl": (-0.1485, 6.049),
            "rl": (-0.1485, 0.873),
            "rr": (2.1485, 0.873),
            "fr": (2.1485, 6.049),
            "center": (1.0, 3.461),
        }

        # We use the private variable for ease of test
        for expected, actual in zip(expected_values.values(), car_footprint._points_of_interest.values()):
            np.testing.assert_array_almost_equal(expected, tuple(actual), 6)  # type: ignore

        # Lastly we check the getter works correctly
        np.testing.assert_array_almost_equal(expected_values['fl'],  # type: ignore
                                             tuple(car_footprint.get_point_of_interest(CarPointType.FRONT_LEFT)), 6)


if __name__ == '__main__':
    unittest.main()
