import base64
import pickle
import tempfile
import unittest
from pathlib import Path

from bokeh.document.document import Document
from bokeh.layouts import LayoutDOM
from bokeh.models import Panel
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.tabs.configuration_tab import ConfigurationTab
from nuplan.planning.nuboard.tabs.histogram_tab import HistogramTab


class TestConfigurationTab(unittest.TestCase):
    """ Test nuboard configuration tab functionality. """

    def setUp(self) -> None:
        """ Set up a configuration tab."""

        self.doc = Document()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(main_path=self.tmp_dir.name,
                                        metric_folder='metrics',
                                        simulation_folder='simulations')

        # Make folders
        metric_path = Path(self.nuboard_file.main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        simulation_path = Path(self.nuboard_file.main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)

        self.nuboard_file_name = Path(self.tmp_dir.name) / ('nuboard_file' + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.histogram_tab = HistogramTab(file_paths=[self.nuboard_file],
                                          doc=self.doc)
        self.configuration_tab = ConfigurationTab(file_paths=[self.nuboard_file],
                                                  doc=self.doc,
                                                  tabs=[self.histogram_tab]
                                                  )

    def test_file_path_on_change(self) -> None:
        """ Test function when the file path is changed. """

        self.configuration_tab._file_paths = []
        self.configuration_tab._file_paths_on_change()
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.options, [])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.options, [])

    def test_add_experiment_file(self) -> None:
        """ Test add experiment file function. """

        attr = 'value'
        old = 'None'
        self.configuration_tab._file_paths = []
        self.configuration_tab._add_experiment_file(attr=attr, old=pickle.dumps(old),
                                                    new=base64.b64encode(pickle.dumps(self.nuboard_file.serialize())))

    def test_remove_button_on_change(self) -> None:
        """ Test if remove button helper works when the remove button is clicked. """

        self.configuration_tab._remove_button_on_click()

    def test_panel(self) -> None:
        """ Test panel properties. """

        self.assertIsInstance(self.configuration_tab.panel, Panel)
        self.assertIsInstance(self.configuration_tab.panel.child, LayoutDOM)

    def tearDown(self) -> None:
        """ Remove temporary folders and files. """

        self.tmp_dir.cleanup()
