import json
import os
import tempfile
import unittest
from pathlib import Path

from bokeh.document.document import Document

from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.metrics.metric_file import MetricFile, MetricFileKey
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.tabs.histogram_tab import HistogramTab


class TestHistogramTab(unittest.TestCase):
    """Test nuboard histogram tab functionality."""

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
        json_file = Path(os.path.dirname(os.path.realpath(__file__))) / "json/test_simulation.json"
        with open(json_file, "r") as f:
            simulation_data = json.load(f)

        # Save to a tmp folder
        save_path = simulation_path / planner_name / scenario_type / log_name / scenario_name
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / "1.json"
        with open(save_file, "w") as f:
            json.dump(simulation_data, f)

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
        statistics = {
            MetricStatisticsType.MAX: Statistic(
                name="ego_max_acceleration", unit="meters_per_second_squared", value=2.0
            ),
            MetricStatisticsType.MIN: Statistic(
                name="ego_min_acceleration", unit="meters_per_second_squared", value=0.0
            ),
            MetricStatisticsType.P90: Statistic(
                name="ego_p90_acceleration", unit="meters_per_second_squared", value=1.0
            ),
            MetricStatisticsType.BOOLEAN: Statistic(name="ego_boolean_acceleration", unit="bool", value=True),
        }
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit="meters_per_second_squared", time_stamps=list(time_stamps), values=list(accel))
        result = MetricStatistics(
            metric_computator="ego_acceleration",
            name="ego_acceleration_statistics",
            statistics=statistics,
            time_series=time_series,
            metric_category="Dynamic",
            metric_score=1.0,
        )

        # Set up dummy metric file
        key = MetricFileKey(
            metric_name="ego_acceleration",
            log_name=log_name,
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            planner_name=planner_name,
        )

        # Set up a dummy metric engine and save the results to a metric file.
        metric_engine = MetricsEngine(main_save_path=metric_path, timestamp=0)

        metric_files = {"ego_acceleration": [MetricFile(key=key, metric_statistics=[result])]}

        metric_engine.write_to_files(metric_files=metric_files)

    def setUp(self) -> None:
        """Set up a histogram tab."""
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

        self.nuboard_file_name = Path(self.tmp_dir.name) / ("nuboard_file" + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)
        self.experiment_file_data = ExperimentFileData(file_paths=[self.nuboard_file])
        self.histogram_tab = HistogramTab(experiment_file_data=self.experiment_file_data, doc=self.doc)

    def test_update_histograms(self) -> None:
        """Test update_histograms works as expected when we update choices."""
        # Update scenario type choices
        self.histogram_tab._scenario_type_multi_choice.value = ["Test"]

        # Update scenario name choices
        self.histogram_tab._metric_name_multi_choice.value = ["ego_acceleration_statistics"]

        self.assertEqual(len(self.histogram_tab.histogram_plots.children), 1)

    def test_file_paths_on_change(self) -> None:
        """Test file_paths_on_change function."""
        new_experiment_file_data = ExperimentFileData(file_paths=[])
        self.histogram_tab.file_paths_on_change(
            experiment_file_data=new_experiment_file_data, experiment_file_active_index=[]
        )
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._scenario_type_multi_choice.options, [])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.value, [])
        self.assertEqual(self.histogram_tab._metric_name_multi_choice.options, [])

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
