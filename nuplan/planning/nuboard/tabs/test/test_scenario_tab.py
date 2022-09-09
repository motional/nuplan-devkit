import unittest

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.tabs.scenario_tab import ScenarioTab
from nuplan.planning.nuboard.tabs.test.skeleton_test_tab import SkeletonTestTab
from nuplan.planning.scenario_builder.test.mock_abstract_scenario_builder import MockAbstractScenarioBuilder


class TestScenarioTab(SkeletonTestTab):
    """Test nuboard scenario tab functionality."""

    def setUp(self) -> None:
        """Set up a scenario tab."""
        super().setUp()
        vehicle_parameters = get_pacifica_parameters()
        scenario_builder = MockAbstractScenarioBuilder()
        self.experiment_file_data = ExperimentFileData(file_paths=[self.nuboard_file])
        self.scenario_tab = ScenarioTab(
            experiment_file_data=self.experiment_file_data,
            scenario_builder=scenario_builder,
            vehicle_parameters=vehicle_parameters,
            doc=self.doc,
        )

    def test_update_scenario(self) -> None:
        """Test functions corresponding to selection changes work as expected."""
        self.scenario_tab.file_paths_on_change(
            experiment_file_data=self.experiment_file_data, experiment_file_active_index=[0]
        )
        # Update scenario type choices
        self.scenario_tab._scalar_scenario_type_select.value = self.scenario_tab._scalar_scenario_type_select.options[1]

        # Update scenario log choices
        self.scenario_tab._scalar_log_name_select.value = self.scenario_tab._scalar_log_name_select.options[1]

        # Update scenario name choices
        self.scenario_tab._scalar_scenario_name_select.value = self.scenario_tab._scalar_scenario_name_select.options[1]
        self.assertEqual(len(self.scenario_tab.simulation_tile_layout.children), 1)
        self.assertEqual(len(self.scenario_tab.time_series_layout.children), 1)

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change function."""
        new_experiment_file_data = ExperimentFileData(file_paths=[])
        self.scenario_tab.file_paths_on_change(
            experiment_file_data=new_experiment_file_data, experiment_file_active_index=[]
        )
        self.assertEqual(self.scenario_tab._scalar_scenario_type_select.value, "")
        self.assertEqual(self.scenario_tab._scalar_scenario_type_select.options, [""])
        self.assertEqual(self.scenario_tab._scalar_scenario_name_select.value, "")
        self.assertEqual(self.scenario_tab._scalar_scenario_name_select.options, [])

    def test_update_scenario_legend(self) -> None:
        """Test functions corresponding to legend selection changes work as expected."""
        self.scenario_tab.file_paths_on_change(
            experiment_file_data=self.experiment_file_data, experiment_file_active_index=[0]
        )
        # Update scenario type choices
        self.scenario_tab._scalar_scenario_type_select.value = self.scenario_tab._scalar_scenario_type_select.options[1]

        # Update scenario log choices
        self.scenario_tab._scalar_log_name_select.value = self.scenario_tab._scalar_log_name_select.options[1]

        # Update scenario name choices
        self.scenario_tab._scalar_scenario_name_select.value = self.scenario_tab._scalar_scenario_name_select.options[1]

        # Update trajectory checkbox
        self.scenario_tab._traj_checkbox_group.active = [0]

        # Update map checkbox
        self.scenario_tab._map_checkbox_group.active = [0, 1, 2]

        # Update object checkbox
        self.scenario_tab._object_checkbox_group.active = [3, 4]

    def test_modal_button_on_click(self) -> None:
        """Test modal button on click function."""
        self.scenario_tab._experiment_file_active_index = [0]

        # Update scenario type choices
        self.scenario_tab._scalar_scenario_type_select.value = self.scenario_tab._scalar_scenario_type_select.options[1]

        # Update scenario log choices
        self.scenario_tab._scalar_log_name_select.value = self.scenario_tab._scalar_log_name_select.options[1]

        # Update scenario name choices
        self.scenario_tab._scalar_scenario_name_select.value = self.scenario_tab._scalar_scenario_name_select.options[1]

        # Click the button
        self.scenario_tab._scenario_modal_query_button_on_click()
        self.assertEqual(self.scenario_tab.planner_checkbox_group.labels, ['SimplePlanner'])
        self.assertIn('ego_acceleration_statistics', self.scenario_tab._time_series_data)

    def test_planner_button_on_click(self) -> None:
        """Test checkbox button in planner."""
        # Update scenarios
        self.scenario_tab._experiment_file_active_index = [0]

        # Update scenario type choices
        self.scenario_tab._scalar_scenario_type_select.value = self.scenario_tab._scalar_scenario_type_select.options[1]
        # Update scenario log choices
        self.scenario_tab._scalar_log_name_select.value = self.scenario_tab._scalar_log_name_select.options[1]

        # Update scenario name choices
        self.scenario_tab._scalar_scenario_name_select.value = self.scenario_tab._scalar_scenario_name_select.options[1]

        # Click the button to update planner checkbox group
        self.scenario_tab._scenario_modal_query_button_on_click()

        # Disable the planner
        self.scenario_tab.planner_checkbox_group.active = []

        self.assertEqual(len(self.scenario_tab.simulation_tile_layout.children), 1)
        self.assertEqual(len(self.scenario_tab.time_series_layout.children), 1)

        # Enable the planner
        self.scenario_tab.planner_checkbox_group.active = [0]
        self.assertEqual(len(self.scenario_tab.simulation_tile_layout.children), 1)
        self.assertEqual(len(self.scenario_tab.time_series_layout.children), 1)

        # Expect to raise index error
        with self.assertRaises(IndexError):
            self.scenario_tab.planner_checkbox_group.active = [1]


if __name__ == "__main__":
    unittest.main()
