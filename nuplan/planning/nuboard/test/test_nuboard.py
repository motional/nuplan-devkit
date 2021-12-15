import tempfile
import unittest
from pathlib import Path

from bokeh.document.document import Document
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.scenario_builder.test.mock_abstract_scenario_builder import MockAbstractScenarioBuilder


class TestNuBoard(unittest.TestCase):
    """ Test nuboard functionality. """

    def setUp(self) -> None:
        """ Set up nuboard a bokeh main page."""

        vehicle_parameters = get_pacifica_parameters()
        self.doc = Document()
        scenario_builder = MockAbstractScenarioBuilder()

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
        main_paths = [str(self.nuboard_file_name)]
        self.nuboard = NuBoard(
            profiler_path=Path(self.tmp_dir.name),
            nuboard_paths=main_paths,
            scenario_builder=scenario_builder,
            metric_categories=['Dynamic', 'Planning'],
            vehicle_parameters=vehicle_parameters
        )

    def test_main_page(self) -> None:
        """ Test if successfully construct a bokeh main page. """

        self.nuboard.main_page(doc=self.doc)
        self.assertEqual(len(self.doc.roots), 3)

    def tearDown(self) -> None:
        """ Remove temporary folders and files. """

        self.tmp_dir.cleanup()
