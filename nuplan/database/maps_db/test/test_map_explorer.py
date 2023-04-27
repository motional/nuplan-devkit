import unittest

from nuplan.database.maps_db.map_api import NuPlanMapWrapper
from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer
from nuplan.database.tests.test_utils_nuplan_db import get_test_maps_db


class TestMapExplorer(unittest.TestCase):
    """Test NuPlanMapExplorer class."""

    def setUp(self) -> None:
        """
        Initialize the map.
        """
        self.maps_db = get_test_maps_db()
        self.location = "us-nv-las-vegas-strip"
        self.nuplan_map = NuPlanMapWrapper(maps_db=self.maps_db, map_name=self.location)
        self.nuplan_explore = NuPlanMapExplorer(self.nuplan_map)

    def test_render_layers(self) -> None:
        """
        Checks the function to render layers.
        """
        try:
            self.nuplan_explore.render_layers(self.nuplan_map.vector_layers, alpha=0.5)
        except RuntimeError:
            self.fail("render_layers() raised RuntimeError unexpectedly!")

    def test_render_map_mask(self) -> None:
        """
        Checks the function to render map mask.
        """
        xmin, ymin, xmax, ymax = self.nuplan_map.get_bounds('lanes_polygons')
        width = xmax - xmin
        height = ymax - ymin
        try:
            self.nuplan_explore.render_map_mask(
                (xmin + (width / 2), ymin + (height / 2), height, width),
                0.0,
                ['lanes_polygons', 'intersections'],
                (500, 500),
                (50, 50),
                2,
            )
        except RuntimeError:
            self.fail("render_map_mask() raised RuntimeError unexpectedly!")

    def test_render_nearby_roads(self) -> None:
        """
        Checks the function to render nearby roads.
        """
        xmin, ymin, xmax, ymax = self.nuplan_map.get_bounds('lanes_polygons')
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + (width / 2) - 921
        y = ymin + (height / 2) + 1540

        try:
            self.nuplan_explore.render_nearby_roads(x, y)
        except RuntimeError:
            self.fail("render_nearby_roads() raised RuntimeError unexpectedly!")


if __name__ == '__main__':
    unittest.main()
