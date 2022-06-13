import unittest

import numpy as np

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.oriented_box import OrientedBoxPointType
from nuplan.common.actor_state.test.test_utils import get_sample_pose
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


class TestCarFootprint(unittest.TestCase):
    """Tests CarFoorprint class"""

    def setUp(self) -> None:
        """Sets sample parameters for testing"""
        self.center_position_from_rear_axle = get_pacifica_parameters().rear_axle_to_center

    def test_car_footprint_creation(self) -> None:
        """Checks that the car footprint is created correctly, in particular the point of interest."""
        car_footprint = CarFootprint.build_from_rear_axle(get_sample_pose(), get_pacifica_parameters())

        self.assertAlmostEqual(car_footprint.rear_axle_to_center_dist, self.center_position_from_rear_axle)

        # Check that the point of interest are created correctly
        expected_values = {
            OrientedBoxPointType.FRONT_BUMPER: (1.0, 6.049),
            OrientedBoxPointType.REAR_BUMPER: (1.0, 0.873),
            OrientedBoxPointType.FRONT_LEFT: (-0.1485, 6.049),
            OrientedBoxPointType.REAR_LEFT: (-0.1485, 0.873),
            OrientedBoxPointType.REAR_RIGHT: (2.1485, 0.873),
            OrientedBoxPointType.FRONT_RIGHT: (2.1485, 6.049),
            OrientedBoxPointType.CENTER: (1.0, 3.461),
        }

        # We use the private variable for ease of test
        for point, position in expected_values.items():
            np.testing.assert_array_almost_equal(position, tuple(car_footprint.corner(point)))

        # Lastly we check the getter works correctly
        np.testing.assert_array_almost_equal(
            expected_values[OrientedBoxPointType.FRONT_LEFT],
            tuple(car_footprint.get_point_of_interest(OrientedBoxPointType.FRONT_LEFT)),
            6,
        )


if __name__ == '__main__':
    unittest.main()
