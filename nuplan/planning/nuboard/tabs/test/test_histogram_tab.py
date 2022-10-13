import unittest

from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.tabs.histogram_tab import HistogramTab
from nuplan.planning.nuboard.tabs.test.skeleton_test_tab import SkeletonTestTab


class TestHistogramTab(SkeletonTestTab):
    """Test nuboard histogram tab functionality."""

    def setUp(self) -> None:
        """Set up a histogram tab."""
        super().setUp()
        self.histogram_tab = HistogramTab(experiment_file_data=self.experiment_file_data, doc=self.doc)

    def test_update_histograms(self) -> None:
        """Test update_histograms works as expected when we update choices."""
        self.histogram_tab.file_paths_on_change(
            experiment_file_data=self.experiment_file_data, experiment_file_active_index=[0]
        )
        # Update scenario type choices
        self.histogram_tab._scenario_type_multi_choice.value = ["Test"]

        # Update scenario name choices
        self.histogram_tab._metric_name_multi_choice.value = ["ego_acceleration_statistics"]
        self.histogram_tab._setting_modal_query_button_on_click()

        self.assertIn('ego_acceleration_statistics', self.histogram_tab._aggregated_data)
        self.assertEqual(len(self.histogram_tab.histogram_plots.children), 1)

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change function."""
        new_experiment_file_data = ExperimentFileData(file_paths=[])
        self.histogram_tab.file_paths_on_change(
            experiment_file_data=new_experiment_file_data, experiment_file_active_index=[]
        )
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.options, ['all'])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.options, [])


if __name__ == "__main__":
    unittest.main()
