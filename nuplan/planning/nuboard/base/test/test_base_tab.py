import tempfile
import unittest
from pathlib import Path

from bokeh.document.document import Document
from bokeh.palettes import Bokeh, Category20, Set3

from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.metrics.metric_file import MetricFile, MetricFileKey
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.utils.test.utils import create_sample_simulation_log
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback


class TestBaseTab(unittest.TestCase):
    """Test base_tab functionality."""

    def set_up_dummy_simulation(
        self,
        simulation_path: Path,
        log_name: str,
        planner_name: str,
        scenario_type: str,
        scenario_name: str,
    ) -> None:
        """
        Set up dummy simulation data.
        :param simulation_path: Simulation path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        # Create a sample SimulationLog and save it to a tmp folder
        save_path = simulation_path / planner_name / scenario_type / log_name / scenario_name
        save_path.mkdir(parents=True, exist_ok=True)
        simulation_data = create_sample_simulation_log(save_path / "test_base_tab_simulation_log.msgpack.xz")

        simulation_data.save_to_file()

    def set_up_dummy_metric(
        self, metric_path: Path, log_name: str, planner_name: str, scenario_type: str, scenario_name: str
    ) -> None:
        """
        Set up dummy metric results.
        :param metric_path: Metric path.
        :param log_name: Log name.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """
        # Set up dummy metric statistics
        statistics = [
            Statistic(
                name="ego_max_acceleration", unit="meters_per_second_squared", value=2.0, type=MetricStatisticsType.MAX
            ),
            Statistic(
                name="ego_min_acceleration", unit="meters_per_second_squared", value=0.0, type=MetricStatisticsType.MIN
            ),
            Statistic(
                name="ego_p90_acceleration", unit="meters_per_second_squared", value=1.0, type=MetricStatisticsType.P90
            ),
        ]
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit="meters_per_second_squared", time_stamps=list(time_stamps), values=list(accel))
        result = MetricStatistics(
            metric_computator="ego_acceleration",
            name="ego_acceleration_statistics",
            statistics=statistics,
            time_series=time_series,
            metric_category="Dynamic",
            metric_score=1,
        )

        # Set up dummy metric file
        key = MetricFileKey(
            metric_name="ego_acceleration",
            scenario_name=scenario_name,
            log_name=log_name,
            scenario_type=scenario_type,
            planner_name=planner_name,
        )

        # Set up a dummy metric engine and save the results to a metric file.
        metric_engine = MetricsEngine(main_save_path=metric_path)

        metric_files = {"ego_acceleration": [MetricFile(key=key, metric_statistics=[result])]}

        metric_engine.write_to_files(metric_files=metric_files)

        # Integrate to a metric file
        metric_file_callback = MetricFileCallback(
            metric_file_output_path=str(metric_path), scenario_metric_paths=[str(metric_path)]
        )
        metric_file_callback.on_run_simulation_end()

    def setUp(self) -> None:
        """Set up a nuboard base tab."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(
            simulation_main_path=self.tmp_dir.name,
            metric_main_path=self.tmp_dir.name,
            metric_folder="metrics",
            simulation_folder="simulations",
            aggregator_metric_folder="aggregator_metric",
            current_path=Path(self.tmp_dir.name),
        )
        doc = Document()
        log_name = 'dummy_log'
        planner_name = "SimplePlanner"
        scenario_type = "Test"
        scenario_name = "Dummy_scene"

        # Set up dummy metric files
        metric_path = Path(self.nuboard_file.metric_main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_metric(
            metric_path=metric_path,
            log_name=log_name,
            planner_name=planner_name,
            scenario_name=scenario_name,
            scenario_type=scenario_type,
        )

        # Set up dummy simulation files
        simulation_path = Path(self.nuboard_file.simulation_main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_simulation(
            simulation_path,
            log_name=log_name,
            planner_name=planner_name,
            scenario_type=scenario_type,
            scenario_name=scenario_name,
        )
        color_palettes = Category20[20] + Set3[12] + Bokeh[8]
        experiment_file_data = ExperimentFileData(file_paths=[], color_palettes=color_palettes)
        self.base_tab = BaseTab(doc=doc, experiment_file_data=experiment_file_data)

    def test_update_experiment_file_data(self) -> None:
        """Test update experiment file data."""
        self.base_tab.experiment_file_data.update_data(file_paths=[self.nuboard_file])
        self.assertEqual(len(self.base_tab.experiment_file_data.available_metric_statistics_names), 1)
        self.assertEqual(len(self.base_tab.experiment_file_data.simulation_scenario_keys), 1)

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change feature."""
        self.base_tab.experiment_file_data.update_data(file_paths=[self.nuboard_file])
        self.assertRaises(
            NotImplementedError, self.base_tab.file_paths_on_change, self.base_tab.experiment_file_data, [0]
        )

    def tearDown(self) -> None:
        """Remove all temporary folders and files."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
