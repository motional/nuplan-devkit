import os
import tempfile
import unittest
from pathlib import Path

from bokeh.document.document import Document

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.metrics.metric_file import MetricFile, MetricFileKey
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.tabs.scenario_tab import ScenarioTab
from nuplan.planning.scenario_builder.test.mock_abstract_scenario_builder import MockAbstractScenarioBuilder
from nuplan.planning.simulation.simulation_log import SimulationLog


class TestScenarioTab(unittest.TestCase):
    """Test nuboard scenario tab functionality."""

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
        simulation_file = (
            Path(os.path.dirname(os.path.realpath(__file__))) / "simulation_log/test_simulation_log.msgpack.xz"
        )
        simulation_data = SimulationLog.load_data(simulation_file)

        # Save to a tmp folder
        save_path = simulation_path / planner_name / scenario_type / log_name / scenario_name
        save_path.mkdir(parents=True, exist_ok=True)
        simulation_data.file_path = save_path / "1.msgpack.xz"
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
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            planner_name=planner_name,
            log_name=log_name,
        )
        metric_files = {"ego_acceleration": [MetricFile(key=key, metric_statistics=[result])]}

        # Set up a dummy metric engine and save the results to a metric file.
        metric_engine = MetricsEngine(main_save_path=metric_path, timestamp=0)
        metric_engine.write_to_files(metric_files=metric_files)

    def setUp(self) -> None:
        """Set up an overview tab."""
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

        log_name = "Dummy_log"
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
            planner_name=planner_name,
            log_name=log_name,
            scenario_type=scenario_type,
            scenario_name=scenario_name,
        )

        self.nuboard_file_name = Path(self.tmp_dir.name) / ("nuboard_file" + self.nuboard_file.extension())
        self.nuboard_file.save_nuboard_file(self.nuboard_file_name)

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
        # Update scenario type choices
        self.scenario_tab._scalar_scenario_type_select.value = "Test"

        # Update scenario name choices
        self.scenario_tab._scalar_scenario_name_select.value = "Dummy_scene"
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

    def tearDown(self) -> None:
        """Remove temporary folders and files."""
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
