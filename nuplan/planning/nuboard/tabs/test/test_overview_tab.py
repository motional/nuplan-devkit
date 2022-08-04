import unittest

from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.tabs.overview_tab import OverviewTab
from nuplan.planning.nuboard.tabs.test.skeleton_test_tab import SkeletonTestTab


class TestOverviewTab(SkeletonTestTab):
    """Test nuboard overview tab functionality."""

    def setUp(self) -> None:
        """Set up an overview tab."""
        super().setUp()
        self.overview_tab = OverviewTab(experiment_file_data=self.experiment_file_data, doc=self.doc)

    def test_update_table(self) -> None:
        """Test update table function."""
        self.overview_tab._overview_on_change()

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change function."""
        new_experiment_file_data = ExperimentFileData(file_paths=[])
        self.overview_tab.file_paths_on_change(
            experiment_file_data=new_experiment_file_data, experiment_file_active_index=[]
        )


if __name__ == "__main__":
    unittest.main()
