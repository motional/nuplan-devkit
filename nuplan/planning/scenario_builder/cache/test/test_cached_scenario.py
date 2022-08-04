import unittest

from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario


class TestCachedScenario(unittest.TestCase):
    """
    Test suite for CachedScenario
    """

    def _make_cached_scenario(self) -> CachedScenario:
        return CachedScenario(log_name="log/name", token="token", scenario_type="type")

    def test_token(self) -> None:
        """
        Test the token method.
        """
        scenario = self._make_cached_scenario()
        self.assertEqual("token", scenario.token)

    def test_log_name(self) -> None:
        """
        Test the log_name method.
        """
        scenario = self._make_cached_scenario()
        self.assertEqual("log/name", scenario.log_name)

    def test_scenario_name_raises(self) -> None:
        """
        Test that the scenario_name method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.scenario_name

    def test_ego_vehicle_parameters_raises(self) -> None:
        """
        Test that the ego_vehicle_parameters method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.ego_vehicle_parameters

    def test_scenario_type(self) -> None:
        """
        Test that the scenario_type method returns scenario_type.
        """
        scenario = self._make_cached_scenario()
        self.assertEqual("type", scenario.scenario_type)

    def test_map_api_raises(self) -> None:
        """
        Test that the map_api method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.map_api

    def test_database_interval_raises(self) -> None:
        """
        Test that the database_interval method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.database_interval

    def test_get_number_of_iterations_raises(self) -> None:
        """
        Test that the get_number_of_iterations method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_number_of_iterations()

    def test_get_time_point_raises(self) -> None:
        """
        Test that the get_time_point method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_time_point()

    def test_get_lidar_to_ego_transform_raises(self) -> None:
        """
        Test that the get_lidar_to_ego_transform method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_lidar_to_ego_transform()

    def test_get_mission_goal_raises(self) -> None:
        """
        Test that the get_mission_goal method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_mission_goal()

    def test_get_route_roadblock_ids_raises(self) -> None:
        """
        Test that the get_route_roadblock_ids method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_route_roadblock_ids()

    def test_get_expert_goal_state_raises(self) -> None:
        """
        Test that the get_expert_goal_state method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_expert_goal_state()

    def test_get_tracked_objects_at_iteration_raises(self) -> None:
        """
        Test that the get_tracked_objects_at_iteration method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_tracked_objects_at_iteration(0)

    def test_get_sensors_at_iteration_raises(self) -> None:
        """
        Test that the get_sensors_at_iteration method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_sensors_at_iteration(0)

    def test_get_ego_state_at_iteration_raises(self) -> None:
        """
        Test that the get_ego_state_at_iteration method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_ego_state_at_iteration(0)

    def test_get_traffic_light_status_at_iteration_raises(self) -> None:
        """
        Test that the get_traffic_light_status_at_iteration method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_traffic_light_status_at_iteration(0)

    def test_get_future_timestamps_raises(self) -> None:
        """
        Test that the get_future_timestamps method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_future_timestamps(0, 0, 0)

    def test_get_past_timestamps_raises(self) -> None:
        """
        Test that the get_past_timestamps method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_past_timestamps(0, 0, 0)

    def test_ego_future_trajectory_raises(self) -> None:
        """
        Test that the ego_future_trajectory method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_ego_future_trajectory(0, 0, 0)

    def test_get_ego_past_trajectory_raises(self) -> None:
        """
        Test that the get_ego_past_trajectory method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_ego_past_trajectory(0, 0, 0)

    def test_get_past_sensors_raises(self) -> None:
        """
        Test that the get_past_sensors method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_past_sensors(0, 0, 0)

    def test_get_past_tracked_objects_raises(self) -> None:
        """
        Test that the get_past_tracked_objects method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_past_tracked_objects(0, 0, 0)

    def test_get_future_tracked_objects_raises(self) -> None:
        """
        Test that the get_future_tracked_objects method raises an error.
        """
        scenario = self._make_cached_scenario()
        with self.assertRaises(NotImplementedError):
            _ = scenario.get_future_tracked_objects(0, 0, 0)


if __name__ == "__main__":
    unittest.main()
