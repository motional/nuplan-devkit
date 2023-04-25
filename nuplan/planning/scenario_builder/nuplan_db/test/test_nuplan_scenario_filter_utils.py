import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import Mock, patch

from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import (
    filter_ego_has_route,
    filter_ego_starts,
    filter_ego_stops,
    filter_fraction_lidarpc_tokens_in_set,
    filter_non_stationary_ego,
    filter_scenarios_by_timestamp,
    filter_total_num_scenarios,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import DEFAULT_SCENARIO_NAME
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import LaneSegmentRoadBlockIDs
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool


class TestNuPlanScenarioFilterUtils(unittest.TestCase):
    """
    Tests scenario filter utils for NuPlan
    """

    def _get_mock_scenario_dict(self) -> Dict[str, List[CachedScenario]]:
        """Gets mock scenario dict."""
        return {
            DEFAULT_SCENARIO_NAME: [
                CachedScenario(log_name="log/name", token=DEFAULT_SCENARIO_NAME, scenario_type=DEFAULT_SCENARIO_NAME)
                for i in range(500)
            ],
            'lane_following_with_lead': [
                CachedScenario(
                    log_name="log/name", token='lane_following_with_lead', scenario_type='lane_following_with_lead'
                )
                for i in range(80)
            ],
            'unprotected_left_turn': [
                CachedScenario(
                    log_name="log/name", token='unprotected_left_turn', scenario_type='unprotected_left_turn'
                )
                for i in range(120)
            ],
        }

    def _get_mock_nuplan_scenario_dict_for_timestamp_filtering(self) -> Dict[str, List[CachedScenario]]:
        """Gets mock scenario dict."""
        mock_scenario_dict = {
            DEFAULT_SCENARIO_NAME: [Mock(NuPlanScenario) for _ in range(0, 100, 3)],
            'lane_following_with_lead': [Mock(NuPlanScenario) for _ in range(0, 100, 6)],
            'lane_following_without_lead': [Mock(NuPlanScenario) for _ in range(3)],
        }

        for i in range(0, len(mock_scenario_dict[DEFAULT_SCENARIO_NAME]) * int(1e6), int(1e6)):
            mock_scenario_dict[DEFAULT_SCENARIO_NAME][int(i / 1e6)]._initial_lidar_timestamp = i * 3
        for i in range(0, len(mock_scenario_dict['lane_following_with_lead']) * int(1e6), int(1e6)):
            mock_scenario_dict['lane_following_with_lead'][int(i / 1e6)]._initial_lidar_timestamp = i * 6

        mock_scenario_dict['lane_following_without_lead'][0]._initial_lidar_timestamp = 5.0 * int(1e6)
        mock_scenario_dict['lane_following_without_lead'][1]._initial_lidar_timestamp = 100.0 * int(1e6)
        mock_scenario_dict['lane_following_without_lead'][2]._initial_lidar_timestamp = 6.0 * int(1e6)

        return mock_scenario_dict

    def _get_mock_worker_map(self) -> Callable[..., List[Any]]:
        """
        Gets mock worker_map function.
        """

        def mock_worker_map(worker: WorkerPool, fn: Callable[..., List[Any]], input_objects: List[Any]) -> List[Any]:
            """
            Mock function for worker_map
            :param worker: Worker pool
            :param fn: Callable function
            :param input_objects: List of objects to be used as input
            :return: List of output objects
            """
            return fn(input_objects)

        return mock_worker_map

    def test_filter_total_num_scenarios_int_max_scenarios_requires_removing_known_scenario_types(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 100
        randomize = True

        final_scenario_dict = filter_total_num_scenarios(
            mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize
        )

        # Assert that default scenarios have been removed from the scenario_dict
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)

        # Assert known scenario types have been removed
        self.assertTrue(
            len(final_scenario_dict['lane_following_with_lead']) < len(mock_scenario_dict['lane_following_with_lead'])
        )
        self.assertTrue(
            len(final_scenario_dict['unprotected_left_turn']) < len(mock_scenario_dict['unprotected_left_turn'])
        )
        self.assertEqual(sum(len(scenarios) for scenarios in final_scenario_dict.values()), limit_total_scenarios)

    def test_filter_total_num_scenarios_int_max_scenarios_less_than_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 300
        randomize = True

        final_scenario_dict = filter_total_num_scenarios(
            mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize
        )

        # Assert that only the default scenarios were removed
        self.assertNotEqual(final_scenario_dict[DEFAULT_SCENARIO_NAME], mock_scenario_dict[DEFAULT_SCENARIO_NAME])
        self.assertEqual(
            final_scenario_dict['lane_following_with_lead'], mock_scenario_dict['lane_following_with_lead']
        )
        self.assertEqual(final_scenario_dict['unprotected_left_turn'], mock_scenario_dict['unprotected_left_turn'])
        self.assertEqual(sum(len(scenarios) for scenarios in final_scenario_dict.values()), limit_total_scenarios)

    def test_filter_total_num_scenarios_int_max_scenarios_more_than_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an int, the actual number of scenarios,
        where the number of scenarios required is less than the total number of scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 800
        randomize = True

        final_scenario_dict = filter_total_num_scenarios(
            mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize
        )

        # Assert no scenarios were removed
        self.assertDictEqual(final_scenario_dict, mock_scenario_dict)

    def test_filter_total_num_scenarios_float_requires_removing_known_scenario_types(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an float, the actual number of scenarios,
        where the number of scenarios required is requires reomving known scenario types.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.2
        randomize = True
        final_num_of_scenarios = int(
            limit_total_scenarios * sum(len(scenarios) for scenarios in mock_scenario_dict.values())
        )
        final_scenario_dict = filter_total_num_scenarios(
            mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize
        )

        # Assert that default scenarios have been removed from the scenario_dict
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)

        # Assert known scenario types have been removed
        self.assertTrue(
            len(final_scenario_dict['lane_following_with_lead']) < len(mock_scenario_dict['lane_following_with_lead'])
        )
        self.assertTrue(
            len(final_scenario_dict['unprotected_left_turn']) < len(mock_scenario_dict['unprotected_left_turn'])
        )
        self.assertEqual(sum(len(scenarios) for scenarios in final_scenario_dict.values()), final_num_of_scenarios)

    def test_filter_total_num_scenarios_float_removes_only_default_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios as an float, the actual number of scenarios,
        where the number of scenarios required is requires reomving known scenario types.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.5
        randomize = True
        final_num_of_scenarios = int(
            limit_total_scenarios * sum(len(scenarios) for scenarios in mock_scenario_dict.values())
        )
        final_scenario_dict = filter_total_num_scenarios(
            mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize
        )

        # Assert that only default scenarios have been removed
        self.assertNotEqual(final_scenario_dict[DEFAULT_SCENARIO_NAME], mock_scenario_dict[DEFAULT_SCENARIO_NAME])

        # Assert known scenario types have not been removed
        self.assertEqual(
            final_scenario_dict['lane_following_with_lead'], mock_scenario_dict['lane_following_with_lead']
        )
        self.assertEqual(final_scenario_dict['unprotected_left_turn'], mock_scenario_dict['unprotected_left_turn'])
        self.assertEqual(sum(len(scenarios) for scenarios in final_scenario_dict.values()), final_num_of_scenarios)

    def test_remove_all_scenarios_int_limit_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to 0. This should raise an assertion error.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0
        randomize = True
        with self.assertRaises(AssertionError):
            filter_total_num_scenarios(
                mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize
            )

    def test_remove_all_scenarios_float_limit_total_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to 0. This should raise an assertion error.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 0.0
        randomize = True
        with self.assertRaises(AssertionError):
            filter_total_num_scenarios(
                mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize
            )

    def test_remove_exactly_all_default_scenarios(self) -> None:
        """
        Tests filter_total_num_scenarios with limit_total_scenarios equal to number of known scenarios.
        """
        mock_scenario_dict = self._get_mock_scenario_dict()
        limit_total_scenarios = 200
        randomize = True
        final_scenario_dict = filter_total_num_scenarios(
            mock_scenario_dict.copy(), limit_total_scenarios=limit_total_scenarios, randomize=randomize
        )

        # Assert that only default scenarios have been removed
        self.assertTrue(DEFAULT_SCENARIO_NAME not in final_scenario_dict)

        # Assert known scenario types have not been removed
        self.assertEqual(
            len(final_scenario_dict['lane_following_with_lead']), len(mock_scenario_dict['lane_following_with_lead'])
        )
        self.assertEqual(
            len(final_scenario_dict['unprotected_left_turn']), len(mock_scenario_dict['unprotected_left_turn'])
        )
        self.assertEqual(sum(len(scenarios) for scenarios in final_scenario_dict.values()), limit_total_scenarios)

    def test_filter_scenarios_by_timestamp(self) -> None:
        """
        Tests filter_scenarios_by_timestamp with default threshold
        """
        mock_worker_map = self._get_mock_worker_map()
        mock_nuplan_scenario_dict = self._get_mock_nuplan_scenario_dict_for_timestamp_filtering()
        with patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.worker_map", mock_worker_map
        ):
            final_scenario_dict = filter_scenarios_by_timestamp(mock_nuplan_scenario_dict.copy())
            # final_scenario_dict['lane_following_with_lead'] bins (bucket to list of initial lidar timestamps) before filtering
            # {5: [0], 10: [6], 15: [12], 20: [18], 25: [24], 35: [30], 40: [36], 45: [42], 50: [48],
            # 55: [54], 65: [60], 70: [66], 75: [72], 80: [78], 85: [84], 95: [90], 100: [96]}
            self.assertEqual(
                len(final_scenario_dict['lane_following_with_lead']),
                len(mock_nuplan_scenario_dict['lane_following_with_lead']),
            )
            # final_scenario_dict[DEFAULT_SCENARIO_NAME] bins (bucket to list of initial lidar timestamps) before filtering
            # {5: [0, 3], 10: [6, 9], 15: [12], 20: [15, 18], 25: [21, 24], 30: [27], 35: [30, 33],
            # 40: [36, 39], 45: [42], 50: [45, 48], 55: [51, 54], 60: [57], 65: [60, 63], 70: [66, 69],
            # 75: [72], 80: [75, 78], 85: [81, 84], 90: [87], 95: [90, 93], 100: [96, 99]}
            self.assertEqual(
                len(final_scenario_dict[DEFAULT_SCENARIO_NAME]),
                len(mock_nuplan_scenario_dict[DEFAULT_SCENARIO_NAME]) * 0.5,
            )
            # check that error is not thrown when there are no scenarios in the bins in between
            self.assertEqual(
                len(final_scenario_dict['lane_following_without_lead']),
                len(mock_nuplan_scenario_dict['lane_following_without_lead']) - 1,
            )

    def test_filter_fraction_lidarpc_tokens_in_set(self) -> None:
        """
        Test filter_fraction_lidarpc_tokens_in_set with fractional thresholds {0, 0.5, 1}.
        """
        alphabet = ["a", "b", "c", "d", "e", "f"]
        mock_nuplan_scenarios = []
        for start_letter in range(4):
            mock_nuplan_scenario = Mock(NuPlanScenario)
            mock_nuplan_scenario.get_scenario_tokens.return_value = set(alphabet[start_letter : start_letter + 3])
            mock_nuplan_scenarios.append(mock_nuplan_scenario)

        (
            full_intersection_scenario,
            two_intersection_scenario,
            one_intersection_scenario,
            no_intersection_scenario,
        ) = mock_nuplan_scenarios

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json_path = Path(tmp_dir) / "tmp_token_set.json"
            json.dump(["a", "b", "c"], open(tmp_json_path, "w"))

            scenario_dict = {"on_pickup_dropoff": [no_intersection_scenario, one_intersection_scenario]}
            self.assertEqual(
                filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 0),
                {"on_pickup_dropoff": [one_intersection_scenario]},
            )
            scenario_dict["on_pickup_dropoff"] = [one_intersection_scenario, two_intersection_scenario]
            self.assertEqual(
                filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 0.5),
                {"on_pickup_dropoff": [two_intersection_scenario]},
            )
            scenario_dict["on_pickup_dropoff"] = [two_intersection_scenario, full_intersection_scenario]
            self.assertEqual(
                filter_fraction_lidarpc_tokens_in_set(scenario_dict, tmp_json_path, 1),
                {"on_pickup_dropoff": [full_intersection_scenario]},
            )

    def test_filter_non_stationary_ego(self) -> None:
        """Test filter_non_stationary_ego with 0.5m displacement threshold"""
        stationary_ego_pudo_scenario = MockAbstractScenario(initial_velocity=StateVector2D(x=0.01, y=0.0))
        mobile_ego_pudo_scenario = MockAbstractScenario()
        scenario_dict = {"on_pickup_dropoff": [stationary_ego_pudo_scenario, mobile_ego_pudo_scenario]}
        filtered_scenario_dict = filter_non_stationary_ego(scenario_dict, minimum_threshold=0.5)
        self.assertEqual(filtered_scenario_dict["on_pickup_dropoff"], [mobile_ego_pudo_scenario])

    def test_filter_ego_starts(self) -> None:
        """Test filter_ego_starts with 0.1 m/s speed threshold"""
        slow_acceleration_scenario = MockAbstractScenario(
            initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=0.01, y=0.0), time_step=1
        )
        fast_acceleration_scenario = MockAbstractScenario(
            initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=1, y=0.0), time_step=1
        )
        scenario_dict = {"on_pickup_dropoff": [slow_acceleration_scenario, fast_acceleration_scenario]}
        filtered_scenario_dict = filter_ego_starts(scenario_dict, speed_threshold=0.1, speed_noise_tolerance=0.1)
        self.assertEqual(filtered_scenario_dict["on_pickup_dropoff"], [fast_acceleration_scenario])

    def test_filter_ego_stops(self) -> None:
        """Test filter_ego_stops with 0.1 m/s speed threshold"""
        slow_deceleration_scenario = MockAbstractScenario(
            initial_velocity=StateVector2D(x=1.0, y=0.0), fixed_acceleration=StateVector2D(x=-0.01, y=0.0), time_step=1
        )
        fast_deceleration_scenario = MockAbstractScenario(
            initial_velocity=StateVector2D(x=1.0, y=0.0), fixed_acceleration=StateVector2D(x=-1 / 9, y=0.0), time_step=1
        )
        scenario_dict = {"on_pickup_dropoff": [slow_deceleration_scenario, fast_deceleration_scenario]}
        filtered_scenario_dict = filter_ego_stops(scenario_dict, speed_threshold=0.1, speed_noise_tolerance=0.1)
        self.assertEqual(filtered_scenario_dict["on_pickup_dropoff"], [fast_deceleration_scenario])

    def test_ego_startstop_noise_tolerance(self) -> None:
        """Test filter_ego_starts with ego barely crossing speed threshold and noise tolerance higher than threshold"""
        fast_enough_acceleration_scenario = MockAbstractScenario(
            initial_velocity=StateVector2D(x=0.0, y=0.0), fixed_acceleration=StateVector2D(x=0.11, y=0.0), time_step=1
        )
        scenario_dict = {"on_pickup_dropoff": [fast_enough_acceleration_scenario]}
        filtered_scenario_dict = filter_ego_starts(scenario_dict, speed_threshold=1, speed_noise_tolerance=2)
        self.assertEqual(filtered_scenario_dict["on_pickup_dropoff"], [])

    def test_filter_ego_has_route(self) -> None:
        """
        Test filter_ego_has_route with one route roadblock in the VectorMap (True case),
        and with no route-intersecting roadblocks (False case).
        """
        map_radius = 35  # Arbitrary for this test; chosen to match default urban driver feature map radius.
        scenario = MockAbstractScenario()
        scenario_dict = {"on_pickup_dropoff": [scenario]}
        with patch(
            'nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_neighbor_vector_map'
        ) as get_neighbor_vector_map:
            get_neighbor_vector_map.return_value = (None, None, None, None, LaneSegmentRoadBlockIDs(['a', 'b', 'c']))
            with patch.object(scenario, 'get_route_roadblock_ids') as get_route_roadblock_ids:
                get_route_roadblock_ids.return_value = ['d', 'e', 'a']
                self.assertEqual(filter_ego_has_route(scenario_dict, map_radius)["on_pickup_dropoff"], [scenario])
                get_route_roadblock_ids.return_value = ['d', 'e', 'f']
                self.assertEqual(filter_ego_has_route(scenario_dict, map_radius)["on_pickup_dropoff"], [])


if __name__ == '__main__':
    unittest.main()
