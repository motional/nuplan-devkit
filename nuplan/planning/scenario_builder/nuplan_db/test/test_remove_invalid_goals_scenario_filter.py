import os
import unittest
from copy import copy

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


class TestRemoveInvalidGoalsFilter(unittest.TestCase):

    def test_remove_invalid_goals_filter(self) -> None:
        """
        Tests that invalid mission goals are correctly filtered out during the database creation.
        """
        scenario_builder = NuPlanScenarioBuilder(
            version="nuplan_v0.1_mini",
            data_root=os.getenv('NUPLAN_DATA_ROOT'))

        filter_with_invalid_goals_args = {
            'log_names': ["2021.07.21.02.32.00_26_1626834838399916.8_1626835894396760.2"],
            'log_labels': None,
            'max_scenarios_per_log': 50,
            'scenario_types': None,
            'scenario_tokens': None,
            'map_name': None,
            'shuffle': False,
            'limit_scenarios_per_type': None,
            'subsample_ratio': None,
            'flatten_scenarios': True,
            'remove_invalid_goals': False,
            'limit_total_scenarios': None,
        }

        filter_with_valid_goals_args = copy(filter_with_invalid_goals_args)
        filter_with_valid_goals_args['remove_invalid_goals'] = True

        worker = Sequential()

        scenarios_with_invalid_goals = scenario_builder.get_scenarios(
            ScenarioFilters(**filter_with_invalid_goals_args), worker)
        scenarios_with_valid_goals = scenario_builder.get_scenarios(
            ScenarioFilters(**filter_with_valid_goals_args), worker)

        scenarios_invalid_goals_removed_tokens = [scenario.token for scenario in scenarios_with_valid_goals]

        for scenario in scenarios_with_invalid_goals:
            if scenario.token not in scenarios_invalid_goals_removed_tokens:
                self.assertIsNone(scenario.get_mission_goal())
            else:
                self.assertIsNotNone(scenario.get_mission_goal())


if __name__ == '__main__':
    unittest.main()
