import unittest

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.training.preprocessing.features.raster_utils import get_agents_raster, get_baseline_paths_raster, \
    get_ego_raster, get_roadmap_raster
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


class TestRasterUtils(unittest.TestCase):

    def setUp(self) -> None:
        """
        Initializes DB
        """
        self.scenario_builder = NuPlanScenarioBuilder(
            version='nuplan_v0.1_mini',
            data_root='/data/sets/nuplan')

        scenario_filter = ScenarioFilters(
            log_names=None,
            log_labels=None,
            max_scenarios_per_log=None,
            scenario_types=None,
            scenario_tokens=None,
            map_name=None,
            shuffle=False,
            limit_scenarios_per_type=None,
            subsample_ratio=0.05,
            flatten_scenarios=True,
            remove_invalid_goals=True,
            limit_total_scenarios=20)

        # Extract scenarios
        worker = Sequential()
        scenarios = self.scenario_builder.get_scenarios(scenario_filter, worker=worker)

        self.x_range = [-56.0, 56.0]
        self.y_range = [-56.0, 56.0]
        self.raster_shape = (224, 224)
        self.resolution = 0.5
        self.thickness = 2

        scenario = scenarios[0]
        self.ego_state = scenario.initial_ego_state
        self.map_api = scenario.map_api
        self.detections = scenario.initial_detections
        self.map_features = {
            'LANE': 255,
            'INTERSECTION': 255,
            'STOP_LINE': 128,
            'CROSSWALK': 128}

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
        roadmap_raster = get_roadmap_raster(
            self.ego_state,
            self.map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.resolution)

        agents_raster = get_agents_raster(
            self.ego_state,
            self.detections,
            self.x_range,
            self.y_range,
            self.raster_shape,
        )

        ego_raster = get_ego_raster(
            self.raster_shape,
            self.ego_longitudinal_offset,
            self.ego_width_pixels,
            self.ego_front_length_pixels,
            self.ego_rear_length_pixels
        )

        baseline_paths_raster = get_baseline_paths_raster(
            self.ego_state,
            self.map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.resolution,
            self.thickness)

        self.assertEqual(roadmap_raster.shape, self.raster_shape)
        self.assertEqual(agents_raster.shape, self.raster_shape)
        self.assertEqual(ego_raster.shape, self.raster_shape)
        self.assertEqual(baseline_paths_raster.shape, self.raster_shape)


if __name__ == '__main__':
    unittest.main()
