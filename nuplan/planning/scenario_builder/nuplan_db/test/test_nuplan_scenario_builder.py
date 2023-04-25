import unittest
from typing import List, Union

import mock

from nuplan.common.utils.test_utils.interface_validation import assert_class_properly_implements_interface
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import (
    GetScenariosFromDbFileParams,
    ScenarioDict,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


class MockNuPlanScenario:
    """
    A dummy NuPlanScenario class to use for unit testing
    """

    def __init__(self, token: str, scenario_type: str) -> None:
        """
        The mock object initialization method.
        :param token: The token to use.
        :param scenario_type: The scneario_type to use.
        """
        self._token = token
        self._scenario_type = scenario_type

    @property
    def token(self) -> str:
        """
        Returns the object's token.
        :return: The token.
        """
        return self._token

    @property
    def scenario_type(self) -> str:
        """
        Returns the object's scenario_type.
        :return: The scenario_type.
        """
        return self._scenario_type


class TestNuPlanScenarioBuilder(unittest.TestCase):
    """
    Tests scenario filtering and construction functionality.
    """

    def test_nuplan_scenario_builder_implements_abstract_scenario_builder(self) -> None:
        """
        Tests that the NuPlanScenarioBuilder implements the AbstractScenarioBuilder interface.
        """
        assert_class_properly_implements_interface(AbstractScenarioBuilder, NuPlanScenarioBuilder)

    def test_get_scenarios_no_filters(self) -> None:
        """
        Tests that the get_scenarios() method functions properly
        With no additional filters applied.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method that validates the input args.
            """
            self.assertIsNone(params.filter_tokens)
            self.assertIsNone(params.filter_types)
            self.assertIsNone(params.filter_map_names)
            self.assertFalse(params.include_cameras)

            m1 = MockNuPlanScenario(token="a", scenario_type="type1")
            m2 = MockNuPlanScenario(token="b", scenario_type="type1")
            m3 = MockNuPlanScenario(token="c", scenario_type="type2")

            return {"type1": [m1, m2], "type2": [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method.
            """
            return ["filename"]

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file",
            db_file_patch,
        ), mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs",
            discover_log_dbs_patch,
        ):
            scenario_builder = NuPlanScenarioBuilder(
                data_root="foo",
                map_root="bar",
                sensor_root="qux",
                db_files=None,
                map_version="baz",
                max_workers=None,
                verbose=False,
                scenario_mapping=None,
                vehicle_parameters=None,
                include_cameras=False,
            )

            scenario_filter = ScenarioFilter(
                scenario_types=None,
                scenario_tokens=None,
                log_names=None,
                map_names=None,
                num_scenarios_per_type=None,
                limit_total_scenarios=None,
                expand_scenarios=False,
                remove_invalid_goals=False,
                shuffle=False,
                timestamp_threshold_s=None,
                ego_displacement_minimum_m=None,
                ego_start_speed_threshold=None,
                ego_stop_speed_threshold=None,
                speed_noise_tolerance=None,
                token_set_path=None,
                fraction_in_token_set_threshold=None,
            )

            result = scenario_builder.get_scenarios(scenario_filter, Sequential())

            self.assertEqual(3, len(result))
            result.sort(key=lambda s: s.token)
            self.assertEqual("a", result[0].token)
            self.assertEqual("b", result[1].token)
            self.assertEqual("c", result[2].token)

    def test_get_scenarios_db_filters(self) -> None:
        """
        Tests that the get_scenarios() method functions properly with db filters applied.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method.
            """
            self.assertEqual(params.filter_tokens, ["a", "b", "c", "d", "e", "f"])
            self.assertEqual(params.filter_types, ["type1", "type2", "type3"])
            self.assertEqual(params.filter_map_names, ["map1", "map2"])
            self.assertTrue(params.include_cameras)

            self.assertTrue(params.log_file_absolute_path in ["filename1", "filename2"])

            m1 = MockNuPlanScenario(token="a", scenario_type="type1")
            m2 = MockNuPlanScenario(token="b", scenario_type="type1")
            m3 = MockNuPlanScenario(token="c", scenario_type="type2")

            return {"type1": [m1, m2], "type2": [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method.
            """
            return ["filename1", "filename2", "filename3"]

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file",
            db_file_patch,
        ), mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs",
            discover_log_dbs_patch,
        ):
            scenario_builder = NuPlanScenarioBuilder(
                data_root="foo",
                map_root="bar",
                sensor_root="qux",
                db_files=None,
                map_version="baz",
                max_workers=None,
                verbose=False,
                scenario_mapping=None,
                vehicle_parameters=None,
                include_cameras=True,
            )

            scenario_filter = ScenarioFilter(
                scenario_types=["type1", "type2", "type3"],
                scenario_tokens=["a", "b", "c", "d", "e", "f"],
                log_names=["filename1", "filename2"],
                map_names=["map1", "map2"],
                num_scenarios_per_type=None,
                limit_total_scenarios=None,
                expand_scenarios=False,
                remove_invalid_goals=False,
                shuffle=False,
                timestamp_threshold_s=None,
                ego_displacement_minimum_m=None,
                ego_start_speed_threshold=None,
                ego_stop_speed_threshold=None,
                speed_noise_tolerance=None,
                token_set_path=None,
                fraction_in_token_set_threshold=None,
            )

            result = scenario_builder.get_scenarios(scenario_filter, Sequential())

            self.assertEqual(6, len(result))
            result.sort(key=lambda s: s.token)
            self.assertEqual("a", result[0].token)
            self.assertEqual("a", result[1].token)
            self.assertEqual("b", result[2].token)
            self.assertEqual("b", result[3].token)
            self.assertEqual("c", result[4].token)
            self.assertEqual("c", result[5].token)

    def test_get_scenarios_num_scenarios_per_type_filter(self) -> None:
        """
        Tests that the get_scenarios() method functions properly
        With a num_scenarios_per_type filter applied.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method
            """
            self.assertEqual(params.filter_tokens, ["a", "b", "c", "d", "e", "f"])
            self.assertEqual(params.filter_types, ["type1", "type2", "type3"])
            self.assertEqual(params.filter_map_names, ["map1", "map2"])
            self.assertEqual(params.include_cameras, False)

            self.assertTrue(params.log_file_absolute_path in ["filename1", "filename2"])

            m1 = MockNuPlanScenario(token="a", scenario_type="type1")
            m2 = MockNuPlanScenario(token="b", scenario_type="type1")
            m3 = MockNuPlanScenario(token="c", scenario_type="type2")

            return {"type1": [m1, m2], "type2": [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method
            """
            return ["filename1", "filename2", "filename3"]

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file",
            db_file_patch,
        ), mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs",
            discover_log_dbs_patch,
        ):
            scenario_builder = NuPlanScenarioBuilder(
                data_root="foo",
                map_root="bar",
                sensor_root="qux",
                db_files=None,
                map_version="baz",
                max_workers=None,
                verbose=False,
                scenario_mapping=None,
                vehicle_parameters=None,
                include_cameras=False,
            )

            scenario_filter = ScenarioFilter(
                scenario_types=["type1", "type2", "type3"],
                scenario_tokens=["a", "b", "c", "d", "e", "f"],
                log_names=["filename1", "filename2"],
                map_names=["map1", "map2"],
                num_scenarios_per_type=2,
                limit_total_scenarios=None,
                expand_scenarios=False,
                remove_invalid_goals=False,
                shuffle=False,
                timestamp_threshold_s=None,
                ego_displacement_minimum_m=None,
                ego_start_speed_threshold=None,
                ego_stop_speed_threshold=None,
                speed_noise_tolerance=None,
                token_set_path=None,
                fraction_in_token_set_threshold=None,
            )

            result = scenario_builder.get_scenarios(scenario_filter, Sequential())

            self.assertEqual(4, len(result))
            self.assertEqual(2, sum(1 if s.scenario_type == "type1" else 0 for s in result))
            self.assertEqual(2, sum(1 if s.scenario_type == "type2" else 0 for s in result))

    def test_get_scenarios_total_num_scenarios_filter(self) -> None:
        """
        Tests that the get_scenarios() method functions properly
        With a total_num_scenarios filter.
        """

        def db_file_patch(params: GetScenariosFromDbFileParams) -> ScenarioDict:
            """
            A patch for the get_scenarios_from_db_file method
            """
            self.assertEqual(params.filter_tokens, ["a", "b", "c", "d", "e", "f"])
            self.assertEqual(params.filter_types, ["type1", "type2", "type3"])
            self.assertEqual(params.filter_map_names, ["map1", "map2"])
            self.assertFalse(params.include_cameras)

            self.assertTrue(params.log_file_absolute_path in ["filename1", "filename2"])

            m1 = MockNuPlanScenario(token="a", scenario_type="type1")
            m2 = MockNuPlanScenario(token="b", scenario_type="type1")
            m3 = MockNuPlanScenario(token="c", scenario_type="type2")

            return {"type1": [m1, m2], "type2": [m3]}

        def discover_log_dbs_patch(load_path: Union[List[str], str]) -> List[str]:
            """
            A patch for the discover_log_dbs method
            """
            return ["filename1", "filename2", "filename3"]

        with mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils.get_scenarios_from_db_file",
            db_file_patch,
        ), mock.patch(
            "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.discover_log_dbs",
            discover_log_dbs_patch,
        ):
            scenario_builder = NuPlanScenarioBuilder(
                data_root="foo",
                map_root="bar",
                sensor_root="qux",
                db_files=None,
                map_version="baz",
                max_workers=None,
                verbose=False,
                scenario_mapping=None,
                vehicle_parameters=None,
                include_cameras=False,
            )

            scenario_filter = ScenarioFilter(
                scenario_types=["type1", "type2", "type3"],
                scenario_tokens=["a", "b", "c", "d", "e", "f"],
                log_names=["filename1", "filename2"],
                map_names=["map1", "map2"],
                num_scenarios_per_type=None,
                limit_total_scenarios=5,
                expand_scenarios=False,
                remove_invalid_goals=False,
                shuffle=False,
                timestamp_threshold_s=None,
                ego_displacement_minimum_m=None,
                ego_start_speed_threshold=None,
                ego_stop_speed_threshold=None,
                speed_noise_tolerance=None,
                token_set_path=None,
                fraction_in_token_set_threshold=None,
            )

            result = scenario_builder.get_scenarios(scenario_filter, Sequential())

            self.assertEqual(5, len(result))


if __name__ == '__main__':
    unittest.main()
