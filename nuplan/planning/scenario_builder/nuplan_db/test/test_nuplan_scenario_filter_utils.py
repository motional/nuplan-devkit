import unittest
from typing import Dict, List

from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import filter_total_num_scenarios
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import DEFAULT_SCENARIO_NAME


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


if __name__ == '__main__':
    unittest.main()
