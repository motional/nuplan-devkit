import random
import unittest
from copy import copy

import numpy as np

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario_builder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


class TestNuPlanScenarioBuilder(unittest.TestCase):
    """
    Tests scenario filtering and construction functionality.
    """

    def setUp(self) -> None:
        """
        Initialize the scenario builder.
        """
        random.seed(0)

        # Objects under test
        self.scenario_builder = get_test_nuplan_scenario_builder()
        self.worker = Sequential()

        # Test parameters
        self.num_scenarios = 5
        self.scenario_types = ['on_pickup_dropoff', 'starting_left_turn']

        # Expected results
        total_samples_in_20s = 400  # 20s @ 20Hz
        total_samples_margin = 5  # +/- 5 samples to account for sample shifts in scenes
        self.min_samples_in_20s = total_samples_in_20s - total_samples_margin
        self.max_samples_in_20s = total_samples_in_20s + total_samples_margin

    def test_all_unknown_single_sample_scenarios(self) -> None:
        """
        Tests filtering of all unknown single-sample scenarios (e.g. used in open-loop training).
        """
        scenario_filter_kwargs = {
            'scenario_types': None,
            'scenario_tokens': None,
            'log_names': None,
            'map_names': None,
            'num_scenarios_per_type': None,
            'limit_total_scenarios': self.num_scenarios,
            'expand_scenarios': True,
            'remove_invalid_goals': False,
            'shuffle': True,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert len(scenarios) == self.num_scenarios
        assert len(scenarios[0]._lidarpc_tokens) == 1

    def test_all_unknown_mutli_sample_scenarios(self) -> None:
        """
        Tests filtering of all unknown multi-sample scenarios (e.g. used in closed-loop training or generic simulation).
        """
        scenario_filter_kwargs = {
            'scenario_types': None,
            'scenario_tokens': None,
            'log_names': None,
            'map_names': None,
            'num_scenarios_per_type': None,
            'limit_total_scenarios': self.num_scenarios,
            'expand_scenarios': False,
            'remove_invalid_goals': False,
            'shuffle': True,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert len(scenarios) == self.num_scenarios
        assert self.min_samples_in_20s < len(scenarios[0]._lidarpc_tokens) < self.max_samples_in_20s

    def test_specific_single_sample_scenario_types(self) -> None:
        """
        Tests filtering of specific single-sample scenario types (e.g. used in open-loop evaluatiion of a model).
        """
        scenario_filter_kwargs = {
            'scenario_types': self.scenario_types,
            'scenario_tokens': None,
            'log_names': None,
            'map_names': None,
            'num_scenarios_per_type': None,
            'limit_total_scenarios': self.num_scenarios,
            'expand_scenarios': True,
            'remove_invalid_goals': False,
            'shuffle': True,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert len(scenarios) == self.num_scenarios
        assert len(scenarios[0]._lidarpc_tokens) == 1

    def test_specific_single_sample_scenario_types_2(self) -> None:
        """
        Tests filtering of a specific single-sample scenario type (e.g. used in open-loop evaluatiion of a model).
        """
        scenario_filter_kwargs = {
            'scenario_types': self.scenario_types,
            'scenario_tokens': None,
            'log_names': None,
            'map_names': None,
            'num_scenarios_per_type': self.num_scenarios,
            'limit_total_scenarios': None,
            'expand_scenarios': True,
            'remove_invalid_goals': False,
            'shuffle': True,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert len(scenarios) == len(self.scenario_types) * self.num_scenarios
        assert len(scenarios[0]._lidarpc_tokens) == 1

        for scenario in scenarios:
            assert scenario.scenario_type in self.scenario_types

    def test_specific_multi_sample_scenario_types(self) -> None:
        """
        Tests filtering of specific multi-sample scenario types (e.g. to compute metrics across all scenario types).
        """
        scenario_filter_kwargs = {
            'scenario_types': self.scenario_types,
            'scenario_tokens': None,
            'log_names': None,
            'map_names': None,
            'num_scenarios_per_type': None,
            'limit_total_scenarios': self.num_scenarios,
            'expand_scenarios': False,
            'remove_invalid_goals': False,
            'shuffle': True,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert len(scenarios) == self.num_scenarios
        assert self.min_samples_in_20s < len(scenarios[0]._lidarpc_tokens) < self.max_samples_in_20s

    def test_scenario_construction_from_token(self) -> None:
        """
        Tests filtering scenarios based on custom token input.
        """
        log_name = '2021.08.31.14.40.58_veh-40_00285_00668'
        token = '97f9f797bc635eb6'

        scenario_filter_kwargs = {
            'scenario_types': None,
            'scenario_tokens': [(log_name, token)],
            'log_names': None,
            'map_names': None,
            'num_scenarios_per_type': None,
            'limit_total_scenarios': None,
            'expand_scenarios': True,
            'remove_invalid_goals': False,
            'shuffle': False,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert len(scenarios) == 1
        assert len(scenarios[0]._lidarpc_tokens) == 1
        assert scenarios[0].token == token
        assert scenarios[0].log_name == log_name

    def test_scenario_filtering_by_log_name(self) -> None:
        """
        Tests filtering scenarios by log name.
        """
        log_name = "2021.07.16.20.45.29_veh-35_01095_01486"

        scenario_filter_kwargs = {
            'scenario_types': None,
            'scenario_tokens': None,
            'log_names': [log_name],
            'map_names': None,
            'num_scenarios_per_type': None,
            'limit_total_scenarios': self.num_scenarios,
            'expand_scenarios': True,
            'remove_invalid_goals': False,
            'shuffle': False,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert len(scenarios) == self.num_scenarios
        assert len(scenarios[0]._lidarpc_tokens) == 1

        for scenario in scenarios:
            assert scenario.log_name == log_name

    def test_scenario_filtering_by_map_name(self) -> None:
        """
        Tests filtering scenarios by map name.
        """
        map_name = "us-nv-las-vegas-strip"

        scenario_filter_kwargs = {
            'scenario_types': None,
            'scenario_tokens': None,
            'log_names': None,
            'map_names': [map_name],
            'num_scenarios_per_type': None,
            'limit_total_scenarios': self.num_scenarios,
            'expand_scenarios': True,
            'remove_invalid_goals': False,
            'shuffle': False,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert len(scenarios) == self.num_scenarios
        assert len(scenarios[0]._lidarpc_tokens) == 1

        for scenario in scenarios:
            assert scenario._initial_lidarpc.log.map_version == map_name

    def test_remove_invalid_goals(self) -> None:
        """
        Tests that invalid mission goals are correctly filtered out.
        """
        filter_with_invalid_goals_kwargs = {
            'scenario_types': None,
            'scenario_tokens': None,
            'log_names': ["2021.07.16.20.45.29_veh-35_01095_01486"],
            'map_names': None,
            'num_scenarios_per_type': None,
            'limit_total_scenarios': 50,
            'expand_scenarios': True,
            'remove_invalid_goals': False,
            'shuffle': False,
        }

        filter_with_valid_goals_args = copy(filter_with_invalid_goals_kwargs)
        filter_with_valid_goals_args['remove_invalid_goals'] = True

        scenarios_with_invalid_goals = self.scenario_builder.get_scenarios(
            ScenarioFilter(**filter_with_invalid_goals_kwargs), self.worker
        )
        scenarios_with_valid_goals = self.scenario_builder.get_scenarios(
            ScenarioFilter(**filter_with_valid_goals_args), self.worker
        )

        scenarios_invalid_goals_removed_tokens = [scenario.token for scenario in scenarios_with_valid_goals]

        for scenario in scenarios_with_invalid_goals:
            if scenario.token not in scenarios_invalid_goals_removed_tokens:
                assert scenario.get_mission_goal() is None
            else:
                assert scenario.get_mission_goal() is not None

    @unittest.skip('We no longer assume that scenarios are sorted by time.')
    def test_limit_scenarios_incrementally(self) -> None:
        """
        Tests that limit scenario filter is applied incrementally on the list of samples.
        """
        log_name = "2021.07.16.20.45.29_veh-35_01095_01486"

        scenario_filter_kwargs = {
            'scenario_types': None,
            'scenario_tokens': None,
            'log_names': [log_name],
            'map_names': None,
            'num_scenarios_per_type': None,
            'limit_total_scenarios': 0.05,
            'expand_scenarios': True,
            'remove_invalid_goals': False,
            'shuffle': False,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert len(scenarios) > 0
        assert len(scenarios[0]._lidarpc_tokens) == 1

        timestamps = [scenario.get_time_point(0).time_us for scenario in scenarios]
        assert (np.diff(timestamps) > 0).all()

    def test_multiple_filters(self) -> None:
        """
        Tests multiple filters simultaneously.
        """
        num_scenarios = 10
        total_factor = 0.5
        log_name = "2021.07.16.20.45.29_veh-35_01095_01486"
        map_name = "us-nv-las-vegas-strip"
        scenario_type = "on_pickup_dropoff"

        scenario_filter_kwargs = {
            'scenario_types': [scenario_type],
            'scenario_tokens': None,
            'log_names': [log_name],
            'map_names': [map_name],
            'num_scenarios_per_type': num_scenarios,
            'limit_total_scenarios': total_factor,
            'expand_scenarios': False,
            'remove_invalid_goals': False,
            'shuffle': True,
        }

        scenarios = self.scenario_builder.get_scenarios(ScenarioFilter(**scenario_filter_kwargs), self.worker)

        assert 0 < len(scenarios) <= int(num_scenarios * total_factor)  # accounting for invalid goals
        assert self.min_samples_in_20s < len(scenarios[0]._lidarpc_tokens) < self.max_samples_in_20s

        for scenario in scenarios:
            assert scenario.scenario_type == scenario_type
            assert scenario._initial_lidarpc.log.map_version == map_name
            assert scenario.log_name == log_name


if __name__ == '__main__':
    unittest.main()
