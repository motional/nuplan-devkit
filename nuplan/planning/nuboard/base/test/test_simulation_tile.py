import tempfile
import unittest
from pathlib import Path

from bokeh.document.document import Document

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.nuboard.base.data_class import NuBoardFile, SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.base.simulation_tile import SimulationTile
from nuplan.planning.nuboard.utils.test.utils import create_sample_simulation_log
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockMapFactory


class TestSimulationTile(unittest.TestCase):
    """Test simulation_tile functionality."""

    def set_up_simulation_log(self, output_path: Path) -> None:
        """
        Create a simulation log and save it to disk.
        :param output path: to write the simulation log to.
        """
        simulation_log = create_sample_simulation_log(output_path)
        simulation_log.save_to_file()

    def setUp(self) -> None:
        """Set up simulation tile with nuboard file."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        vehicle_parameters = get_pacifica_parameters()
        simulation_log_path = Path(self.tmp_dir.name) / "test_simulation_tile_simulation_log.msgpack.xz"
        self.set_up_simulation_log(simulation_log_path)
        nuboard_file = NuBoardFile(
            simulation_main_path=self.tmp_dir.name,
            metric_main_path=self.tmp_dir.name,
            metric_folder="metrics",
            simulation_folder="simulation",
            aggregator_metric_folder="aggregator_metric",
            current_path=Path(self.tmp_dir.name),
        )
        self.scenario_keys = [
            SimulationScenarioKey(
                nuboard_file_index=0,
                log_name='dummy_log',
                planner_name="SimplePlanner",
                scenario_type="common",
                scenario_name="test",
                files=[simulation_log_path],
            )
        ]
        doc = Document()
        map_factory = MockMapFactory()
        experiment_file_data = ExperimentFileData(file_paths=[nuboard_file])
        self.simulation_tile = SimulationTile(
            doc=doc,
            map_factory=map_factory,
            vehicle_parameters=vehicle_parameters,
            radius=80,
            experiment_file_data=experiment_file_data,
        )

    def test_simulation_tile_layout(self) -> None:
        """Test layout design."""
        layout = self.simulation_tile.render_simulation_tiles(
            selected_scenario_keys=self.scenario_keys, figure_sizes=[550, 550]
        )
        self.assertEqual(len(layout), 1)

    def tearDown(self) -> None:
        """Clean up temporary folder and files."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
