import os
import tempfile
import unittest
from pathlib import Path

from bokeh.document.document import Document
from hypothesis import given
from hypothesis import strategies as st

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.scenario_builder.test.mock_abstract_scenario_builder import MockAbstractScenarioBuilder


class TestNuBoard(unittest.TestCase):
    """Test nuboard functionality."""

    def setUp(self) -> None:
        """Set up nuboard a bokeh main page."""
        self.vehicle_parameters = get_pacifica_parameters()
        self.doc = Document()
        self.scenario_builder = MockAbstractScenarioBuilder()

        self.tmp_dir = tempfile.TemporaryDirectory()
        if not os.getenv("NUPLAN_EXP_ROOT", None):
            os.environ["NUPLAN_EXP_ROOT"] = self.tmp_dir.name
        self.nuboard_file = NuBoardFile(
            simulation_main_path=self.tmp_dir.name,
            metric_main_path=self.tmp_dir.name,
            metric_folder="metrics",
            simulation_folder="simulations",
            aggregator_metric_folder="aggregator_metric",
        )

        # Make folders
        metric_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        simulation_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)

        self.nuboard_file_name = Path(self.tmp_dir.name) / ("nuboard_file" + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.main_paths = [str(self.nuboard_file_name)]
        self.nuboard = NuBoard(
            profiler_path=Path(self.tmp_dir.name),
            nuboard_paths=self.main_paths,
            scenario_builder=self.scenario_builder,
            vehicle_parameters=self.vehicle_parameters,
        )

    def test_main_page(self) -> None:
        """Test if successfully construct a bokeh main page."""
        self.nuboard.main_page(doc=self.doc)
        # Number of elements in the main page, change if we add more elements
        self.assertEqual(len(self.doc.roots), 34)

    @given(frame_rate_cap=st.integers(min_value=1, max_value=60))
    def test_valid_frame_rate_cap_range(self, frame_rate_cap: int) -> None:
        """Tests valid frame rate cap range."""
        # No exceptions should be raised
        NuBoard(
            profiler_path=Path(self.tmp_dir.name),
            nuboard_paths=self.main_paths,
            scenario_builder=self.scenario_builder,
            vehicle_parameters=self.vehicle_parameters,
            scenario_rendering_frame_rate_cap_hz=frame_rate_cap,
        )

    @given(frame_rate_cap=st.integers().filter(lambda x: x < 1 or x > 60))
    def test_invalid_frame_rate_cap_range(self, frame_rate_cap: int) -> None:
        """Tests invalid frame rate cap range."""
        with self.assertRaises(ValueError):
            NuBoard(
                profiler_path=Path(self.tmp_dir.name),
                nuboard_paths=self.main_paths,
                scenario_builder=self.scenario_builder,
                vehicle_parameters=self.vehicle_parameters,
                scenario_rendering_frame_rate_cap_hz=frame_rate_cap,
            )

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
