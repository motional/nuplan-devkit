import unittest

from shapely.geometry import LineString, Polygon

from nuplan.planning.simulation.occupancy_map.geopandas_occupancy_map import GeoPandasOccupancyMapFactory, OccupancyMap
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory


class OccupancyMapTests(unittest.TestCase):
    """Tests implementation of OccupancyMap"""

    def setUp(self) -> None:
        """Test setup"""
        self.p1 = Polygon([(0, 0), (0, 2), (3, 2), (3, 0)])
        self.p2 = Polygon([(2, 0), (2, 4), (3, 4), (3, 0)])
        self.p3 = Polygon([(4, 0), (4, 2), (5, 2), (5, 0)])
        self.p4 = Polygon([(0, 4), (0, 5), (1.5, 5), (1.5, 4)])
        self.l1 = LineString([(0, 3), (4, 3), (4, 1)])

    def test_intersects_polygon(self):  # type: ignore
        """Tests polygon-polygon intersections correctness"""
        gp_occupancy_map = GeoPandasOccupancyMapFactory.get_from_geometry([self.p1])
        strtree_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry([self.p1])

        def test(occupancy_map: OccupancyMap) -> None:
            intersection = occupancy_map.intersects(self.p2)
            assert not intersection.is_empty()
            intersection = occupancy_map.intersects(self.p3)
            assert intersection.is_empty()

        test(gp_occupancy_map)
        test(strtree_occupancy_map)

    def test_intersects_linestring(self):  # type: ignore
        """Tests polygon-linestring intersections correctness"""
        gp_occupancy_map = GeoPandasOccupancyMapFactory.get_from_geometry([self.p1])
        strtree_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry([self.p1])

        def test(occupancy_map: OccupancyMap) -> None:
            intersection = occupancy_map.intersects(self.l1)
            assert intersection.is_empty()
            occupancy_map.insert("2", self.p2)
            intersection = occupancy_map.intersects(self.l1)
            assert not intersection.is_empty()

        test(gp_occupancy_map)
        test(strtree_occupancy_map)

    def test_intersects_linestring_buffered(self):  # type: ignore
        """Tests polygon-buffered linestring intersections correctness"""
        gp_occupancy_map = GeoPandasOccupancyMapFactory.get_from_geometry([self.p1])
        strtree_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry([self.p1])

        def test(occupancy_map: OccupancyMap) -> None:
            intersection = occupancy_map.intersects(self.l1.buffer(0.4, cap_style=2))
            assert intersection.is_empty()
            occupancy_map.insert("4", self.p4.buffer(0.5, cap_style=2))
            intersection = occupancy_map.intersects(self.l1.buffer(0.1, cap_style=2))
            assert intersection.is_empty()
            occupancy_map.insert("2", self.p2)
            intersection = occupancy_map.intersects(self.l1.buffer(0.4, cap_style=2))
            assert not intersection.is_empty()

        test(gp_occupancy_map)
        test(strtree_occupancy_map)

    def test_insert_get_set(self):  # type: ignore
        """Tests the expected behavior of get and set"""
        gp_occupancy_map = GeoPandasOccupancyMapFactory.get_from_geometry([self.p1])
        strtree_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry([self.p1])

        def test(occupancy_map: OccupancyMap) -> None:
            assert occupancy_map.size == 1
            occupancy_map.insert("2", self.p3)
            assert occupancy_map.size == 2
            assert self.p3 == occupancy_map.get("2")
            occupancy_map.set("2", self.p2)
            assert self.p2 == occupancy_map.get("2")

        test(gp_occupancy_map)
        test(strtree_occupancy_map)

    def test_get_nearest_entry(self):  # type: ignore
        """Tests expected behavior of get_nearest_entry"""
        gp_occupancy_map = GeoPandasOccupancyMapFactory.get_from_geometry([self.p2, self.p3, self.p4])  # index 0, 1, 2
        strtree_occupancy_map = GeoPandasOccupancyMapFactory.get_from_geometry(
            [self.p2, self.p3, self.p4]
        )  # index 0, 1, 2

        def test(occupancy_map: OccupancyMap) -> None:
            nearest_id, nearest_polygon, distance = occupancy_map.get_nearest_entry_to("0")
            self.assertEqual(nearest_id, "2")
            self.assertEqual(nearest_polygon, self.p4)
            self.assertEqual(distance, 0.5)

        test(gp_occupancy_map)
        test(strtree_occupancy_map)

    def test_get_all(self):  # type: ignore
        """Tests the expected behavior of get_all_ids"""
        gp_occupancy_map = GeoPandasOccupancyMapFactory.get_from_geometry([self.p2, self.p3, self.p4])  # index 0, 1, 2
        strtree_occupancy_map = GeoPandasOccupancyMapFactory.get_from_geometry(
            [self.p2, self.p3, self.p4]
        )  # index 0, 1, 2

        def test(occupancy_map: OccupancyMap) -> None:
            ids = occupancy_map.get_all_ids()
            assert set(ids) == {"0", "1", "2"}
            assert set(ids) != {"0", "1", "3"}

            geoms = occupancy_map.get_all_geometries()

            for actual, expect in zip(geoms, [self.p2, self.p3, self.p4]):
                assert actual == expect

        test(gp_occupancy_map)
        test(strtree_occupancy_map)


if __name__ == '__main__':
    unittest.main()
