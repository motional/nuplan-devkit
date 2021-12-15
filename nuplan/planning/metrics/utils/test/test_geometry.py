import math
import unittest

import nuplan.planning.metrics.utils.geometry as g
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon


class TestSignedLongitudinalAndLateralDistances(unittest.TestCase):

    def setUp(self) -> None:
        """ Creates polygons and poses for testing. """
        self.se2_pose = StateSE2(0.0, 0.0, math.pi / 6)
        self.polygon_base = Polygon([[-2, -1], [2, -1], [2, 1], [-2, 1]])
        self.polygons = [
            self.polygon_base,
            translate(self.polygon_base, xoff=10),
            translate(self.polygon_base, xoff=-10),
            translate(self.polygon_base, yoff=10),
            translate(self.polygon_base, yoff=-10),
            translate(rotate(self.polygon_base, math.pi / 7, use_radians=True), xoff=10, yoff=10)
        ]

        self.expected_lateral_distances = [0.0, -1.99, 1.99, 5.65, -5.65, 1.37]
        self.expected_longitudinal_distances = [0.0, 3.84, -3.84, 0.18, -0.18, 9.0]

    def test_signed_lateral_distance(self) -> None:
        """ Checks signed lateral distance is correct """
        for idx, (polygon, expected_lat_dist) in enumerate(zip(self.polygons, self.expected_lateral_distances)):
            msg = f"Wrong lateral distance on case {idx}"
            self.assertAlmostEqual(g.signed_lateral_distance(self.se2_pose, polygon), expected_lat_dist, 2, msg=msg)

    def test_signed_longitudinal_distance(self) -> None:
        """ Checks signed longitudinal distance projection is correct """
        for idx, (polygon, expected_lon_dist) in enumerate(zip(self.polygons, self.expected_longitudinal_distances)):
            msg = f"Wrong longitudinal distance on case {idx}"
            self.assertAlmostEqual(g.signed_longitudinal_distance(self.se2_pose, polygon), expected_lon_dist, 2,
                                   msg=msg)


class TestLongitudinalAndLateralProjections(unittest.TestCase):
    def setUp(self) -> None:
        """ Creates poses and a point for testing. """
        self.se2_poses = [
            StateSE2(0.0, 0.0, 0.0),
            StateSE2(0.0, 0.0, math.pi / 2),
            StateSE2(1.2, 5.1, 0.17)]

        self.query_point = Point2D(1.36, 3.23)

        self.expected_lateral_distances = [3.23, -1.36, -1.87]
        self.expected_longitudinal_distances = [1.36, 3.23, -0.16]

    def test_lateral_distance(self) -> None:
        """ Checks lateral distance projection is correct """
        for idx, (pose, expected_lat_dist) in enumerate(zip(self.se2_poses, self.expected_lateral_distances)):
            msg = f"Wrong lateral distance on case {idx}"
            self.assertAlmostEqual(g.lateral_distance(pose, self.query_point), expected_lat_dist, 2, msg=msg)

    def test_longitudinal_distance(self) -> None:
        """ Checks longitudinal distance projection is correct """
        for idx, (pose, expected_lon_dist) in enumerate(zip(self.se2_poses, self.expected_longitudinal_distances)):
            msg = f"Wrong longitudinal distance on case {idx}"
            self.assertAlmostEqual(g.longitudinal_distance(pose, self.query_point), expected_lon_dist, 2,
                                   msg=msg)

            if __name__ == '__main__':
                unittest.main()
