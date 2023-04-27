import unittest

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon

from nuplan.database.maps_db.map_api import NuPlanMapWrapper
from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db


class TestMapApi(unittest.TestCase):
    """Test NuPlanMapWrapper class."""

    def setUp(self) -> None:
        """
        Initialize the map for each location.
        """
        self.maps_db = get_test_maps_db()
        self.locations = ["sg-one-north", "us-ma-boston", "us-nv-las-vegas-strip", "us-pa-pittsburgh-hazelwood"]
        self.available_locations = self.maps_db.get_locations()
        self.nuplan_maps = dict()
        for location in self.available_locations:
            self.nuplan_maps[location] = NuPlanMapWrapper(maps_db=self.maps_db, map_name=location)

    def test_version_names(self) -> None:
        """Tests the locations map version are correct."""
        assert len(self.maps_db.version_names) == len(self.available_locations), "Incorrect number of version names"

    def test_locations(self) -> None:
        """
        Checks if maps for all locations are available.
        """
        assert len(self.locations) == len(self.available_locations), "Incorrect number of locations"
        assert sorted(self.locations) == sorted(self.available_locations), "Missing Locations"

    def test_patch_coord(self) -> None:
        """
        Checks the function to get patch coordinates without rotation.
        """
        path_center = [0, 0]
        path_dimension = [10, 10]
        polygon_coords = self.nuplan_maps[self.locations[0]].get_patch_coord(
            (path_center[0], path_center[1], path_dimension[0], path_dimension[1]), 0.0
        )
        expected_polygon_coords = Polygon([[5, -5], [5, 5], [-5, 5], [-5, -5], [5, -5]])
        self.assertEqual(polygon_coords, expected_polygon_coords)

    def test_patch_coord_rotated(self) -> None:
        """
        Checks the function to get patch coordinates with rotation.
        """
        path_center = [0, 0]
        path_dimension = [10, 20]
        polygon_coords = self.nuplan_maps[self.locations[0]].get_patch_coord(
            (path_center[0], path_center[1], path_dimension[0], path_dimension[1]), 90.0
        )
        expected_polygon_coords = Polygon([[5, 10], [-5, 10], [-5, -10], [5, -10], [5, 10]])
        self.assertEqual(polygon_coords, expected_polygon_coords)

    def test_vector_dimensions(self) -> None:
        """
        Checks dimensions of vector layer. It must be less than or equal to size of map.
        """
        for location in self.locations:
            vector_layer_bounds = self.nuplan_maps[location].get_bounds('lanes_polygons')
            map_shape = self.nuplan_maps[location].get_map_dimension()
            self.assertLess(vector_layer_bounds[0], vector_layer_bounds[2])
            self.assertLess(vector_layer_bounds[1], vector_layer_bounds[3])
            self.assertLess(vector_layer_bounds[2] - vector_layer_bounds[0], map_shape[0])
            self.assertLess(vector_layer_bounds[3] - vector_layer_bounds[1], map_shape[1])

    def test_line_in_patch(self) -> None:
        """
        Checks if the line inside patch.
        """
        line_coords = LineString([(1.0, 1.0), (10.0, 10.0)])
        box_coords = [0.0, 0.0, 11.0, 11.0]
        self.assertTrue(self.nuplan_maps[self.locations[0]]._is_line_record_in_patch(line_coords, box_coords))

        box_coords = [0.0, 0.0, 8.0, 8.0]
        self.assertFalse(self.nuplan_maps[self.locations[0]]._is_line_record_in_patch(line_coords, box_coords))

    def test_line_intersects_patch(self) -> None:
        """
        Checks if line intersects the patch.
        """
        line_coords = LineString([(0.0, 0.0), (10.0, 10.0)])
        box_coords = [0.0, 0.0, 11.0, 11.0]
        self.assertTrue(
            self.nuplan_maps[self.locations[0]]._is_line_record_in_patch(line_coords, box_coords, 'intersect')
        )

        box_coords = [11.0, 11.0, 16.0, 16.0]
        self.assertFalse(
            self.nuplan_maps[self.locations[0]]._is_line_record_in_patch(line_coords, box_coords, 'intersect')
        )

    def test_polygon_in_patch(self) -> None:
        """
        Checks if polygon is inside patch.
        """
        polygon_coords = Polygon([(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (1.0, 1.0)])
        box_coords = [0.0, 0.0, 11.0, 11.0]
        self.assertTrue(self.nuplan_maps[self.locations[0]]._is_polygon_record_in_patch(polygon_coords, box_coords))

        box_coords = [0.0, 0.0, 8.0, 8.0]
        self.assertFalse(self.nuplan_maps[self.locations[0]]._is_polygon_record_in_patch(polygon_coords, box_coords))

    def test_polygon_intersects_patch(self) -> None:
        """
        Check if polygon intersects patch.
        """
        polygon_coords = Polygon([(1.0, 1.0), (1.0, 10.0), (10.0, 10.0), (1.0, 1.0)])
        box_coords = [1.0, 1.0, 11.0, 11.0]
        self.assertTrue(
            self.nuplan_maps[self.locations[0]]._is_polygon_record_in_patch(polygon_coords, box_coords, 'intersect')
        )

        box_coords = [12.0, 14.0, 15.0, 15.0]
        self.assertFalse(
            self.nuplan_maps[self.locations[0]]._is_polygon_record_in_patch(polygon_coords, box_coords, 'intersect')
        )

    def test_mask_for_polygons(self) -> None:
        """
        Checks the mask generated using polygons.
        """
        polygon_coords = MultiPolygon([Polygon([(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0), (0.0, 0.0)])])
        mask = np.zeros((10, 10))

        map_explorer = NuPlanMapExplorer(self.nuplan_maps[self.locations[0]])
        predicted_mask = map_explorer.mask_for_polygons(polygon_coords, mask)

        expected_mask = np.zeros((10, 10))
        expected_mask[0:3, 0:3] = 1
        np.testing.assert_array_equal(predicted_mask, expected_mask)

    def test_mask_for_lines(self) -> None:
        """Checks the mask generated using lines."""
        line_coords = LineString([(0, 0), (0, 5), (5, 5), (5, 0), (0, 0)])
        mask = np.zeros((10, 10))

        map_explorer = NuPlanMapExplorer(self.nuplan_maps[self.locations[0]])
        predicted_mask = map_explorer.mask_for_lines(line_coords, mask)

        expected_mask = np.zeros((10, 10))
        expected_mask[0:7, 0:7] = 1
        expected_mask[2:4, 2:4] = 0
        expected_mask[6, 6] = 0
        np.testing.assert_array_equal(predicted_mask, expected_mask)

    def test_layers_on_points(self) -> None:
        """
        Checks if returns correct layers given a point.
        """
        # Raises if layer for checking is a not polygon layer.
        with self.assertRaises(Exception):
            self.nuplan_maps[self.locations[3]].layers_on_point(0, 0, ['lane_connectors'])

        # Checks the output for empty layer input is also empty.
        self.assertFalse(self.nuplan_maps[self.locations[3]].layers_on_point(0, 0, []))

        # check return correct record for a point on the layer.
        layer = self.nuplan_maps[self.locations[2]].layers_on_point(664777.776, 3999698.364, ['lanes_polygons'])
        self.assertEqual(layer['lanes_polygons'], ['63085'])

        # Test return 0 records for a point outside of the layer.
        layer = self.nuplan_maps[self.locations[3]].layers_on_point(87488.0, 43600.0, ['lanes_polygons'])
        self.assertFalse(layer['lanes_polygons'])

    def test_get_records_in_patch(self) -> None:
        """
        Checks the function of getting all the record token that intersects or within a particular rectangular patch.
        """
        # Raises for non vector layer.
        with self.assertRaises(Exception):
            self.nuplan_maps[self.locations[3]].get_records_in_patch([0, 0, 0, 0], ['drivable_area'])

        # Test returning empty tokens in a empty patch.
        tokens = self.nuplan_maps[self.locations[3]].get_records_in_patch([0, 0, 0, 0], ['lanes_polygons'])
        self.assertFalse(tokens['lanes_polygons'])

        # Test returning tokens for a patch.
        xmin, ymin, xmax, ymax = self.nuplan_maps[self.locations[3]].get_bounds('lanes_polygons')
        tokens = self.nuplan_maps[self.locations[3]].get_records_in_patch([xmin, ymin, xmax, ymax], ['lanes_polygons'])
        self.assertTrue(tokens['lanes_polygons'])

    def test_get_layer_polygon(self) -> None:
        """Checks the function of retrieving the polygons of a particular layer within the specified patch."""
        # Raises for non vector layer.
        with self.assertRaises(Exception):
            self.nuplan_maps[self.locations[3]].get_layer_polygon((0, 0, 0, 0), 0.0, 'drivable_area')

        # Test returning empty polygons in a empty patch.
        self.assertFalse(self.nuplan_maps[self.locations[3]].get_layer_polygon((0, 0, 0, 0), 0.0, 'lanes_polygons'))

        # Test returning polygons for a patch.
        xmin, ymin, xmax, ymax = self.nuplan_maps[self.locations[0]].get_bounds('lanes_polygons')
        width = xmax - xmin
        height = ymax - ymin
        patch_box = (xmin + (width / 2), ymin + (height / 2), height, width)
        patch_angle = 0.0
        self.assertTrue(self.nuplan_maps[self.locations[0]].get_layer_polygon(patch_box, patch_angle, 'lanes_polygons'))

    def test_get_layer_line(self) -> None:
        """Checks the function of retrieving the lines of a particular layer within the specified patch."""
        # Raises for non vector layer.
        with self.assertRaises(Exception):
            self.nuplan_maps[self.locations[3]].get_layer_line((0, 0, 0, 0), 0.0, 'drivable_area')

        # Test returning empty lines in an empty patch.
        self.assertFalse(self.nuplan_maps[self.locations[3]].get_layer_line((0, 0, 0, 0), 0.0, 'lanes_polygons'))

        xmin, ymin, xmax, ymax = self.nuplan_maps[self.locations[0]].get_bounds('lanes_polygons')
        width = xmax - xmin
        height = ymax - ymin
        patch_box = (xmin + (width / 2), ymin + (height / 2), height, width)
        patch_angle = 0.0
        self.assertTrue(self.nuplan_maps[self.locations[0]].get_layer_line(patch_box, patch_angle, 'lanes_polygons'))


if __name__ == '__main__':
    unittest.main()
