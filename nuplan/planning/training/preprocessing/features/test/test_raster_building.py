import unittest

import numpy as np

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.training.preprocessing.features.raster_utils import (
    get_agents_raster,
    get_baseline_paths_raster,
    get_ego_raster,
    get_roadmap_raster,
)


class TestRasterUtils(unittest.TestCase):
    """Test raster building utility functions."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        scenario = get_test_nuplan_scenario()

        self.x_range = [-56.0, 56.0]
        self.y_range = [-56.0, 56.0]
        self.raster_shape = (224, 224)
        self.resolution = 0.5
        self.thickness = 2

        self.ego_state = scenario.initial_ego_state
        self.map_api = scenario.map_api
        self.tracked_objects = scenario.initial_tracked_objects
        self.map_features = {'LANE': 255, 'INTERSECTION': 255, 'STOP_LINE': 128, 'CROSSWALK': 128}

        ego_width = 2.297
        ego_front_length = 4.049
        ego_rear_length = 1.127
        self.ego_longitudinal_offset = 0.0
        self.ego_width_pixels = int(ego_width / self.resolution)
        self.ego_front_length_pixels = int(ego_front_length / self.resolution)
        self.ego_rear_length_pixels = int(ego_rear_length / self.resolution)

    def test_get_roadmap_raster(self) -> None:
        """
        Test get_roadmap_raster / get_agents_raster / get_baseline_paths_raster
        """
        # Check if there are tracks in the scene in the first place
        self.assertGreater(len(self.tracked_objects.tracked_objects), 0)

        roadmap_raster = get_roadmap_raster(
            self.ego_state,
            self.map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.resolution,
        )

        agents_raster = get_agents_raster(
            self.ego_state,
            self.tracked_objects,
            self.x_range,
            self.y_range,
            self.raster_shape,
        )

        ego_raster = get_ego_raster(
            self.raster_shape,
            self.ego_longitudinal_offset,
            self.ego_width_pixels,
            self.ego_front_length_pixels,
            self.ego_rear_length_pixels,
        )

        baseline_paths_raster = get_baseline_paths_raster(
            self.ego_state, self.map_api, self.x_range, self.y_range, self.raster_shape, self.resolution, self.thickness
        )

        # Check dimensions
        self.assertEqual(roadmap_raster.shape, self.raster_shape)
        self.assertEqual(agents_raster.shape, self.raster_shape)
        self.assertEqual(ego_raster.shape, self.raster_shape)
        self.assertEqual(baseline_paths_raster.shape, self.raster_shape)

        # Check if objects are drawn on to the raster
        self.assertTrue(np.any(roadmap_raster))
        self.assertTrue(np.any(agents_raster))
        self.assertTrue(np.any(ego_raster))
        self.assertTrue(np.any(baseline_paths_raster))


if __name__ == '__main__':
    unittest.main()
