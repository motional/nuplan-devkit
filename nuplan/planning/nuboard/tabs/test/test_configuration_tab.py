import base64
import pickle
import tempfile
import unittest
from pathlib import Path

from bokeh.document.document import Document

from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.tabs.configuration_tab import ConfigurationTab
from nuplan.planning.nuboard.tabs.histogram_tab import HistogramTab


class TestConfigurationTab(unittest.TestCase):
    """Test nuboard configuration tab functionality."""

    def setUp(self) -> None:
        """Set up a configuration tab."""
        self.doc = Document()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(
            simulation_main_path=self.tmp_dir.name,
            metric_main_path=self.tmp_dir.name,
            metric_folder="metrics",
            simulation_folder="simulations",
            aggregator_metric_folder="aggregator_metric",
            current_path=Path(self.tmp_dir.name),
        )

        # Make folders
        metric_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        simulation_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)

        self.nuboard_file_name = Path(self.tmp_dir.name) / ("nuboard_file" + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.experiment_file_data = ExperimentFileData(file_paths=[self.nuboard_file])
        self.histogram_tab = HistogramTab(experiment_file_data=self.experiment_file_data, doc=self.doc)
        self.configuration_tab = ConfigurationTab(
            experiment_file_data=self.experiment_file_data, doc=self.doc, tabs=[self.histogram_tab]
        )

    def test_file_path_on_change(self) -> None:
        """Test function when the file path is changed."""
        self.configuration_tab._file_paths = []
        self.configuration_tab._file_paths_on_change()
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.options, ['all'])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.options, [])

    def test_add_experiment_file(self) -> None:
        """Test add experiment file function."""
        attr = "value"
        old = "None"
        self.configuration_tab.experiment_file_data.file_paths = []
        self.configuration_tab._add_experiment_file(
            attr=attr, old=pickle.dumps(old), new=base64.b64encode(pickle.dumps(self.nuboard_file.serialize()))
        )

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
