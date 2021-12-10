import json
import os
import tempfile
import unittest
from pathlib import Path

from bokeh.document.document import Document
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.metrics.metric_file import MetricFile, MetricFileKey
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.data_class import NuBoardFile


class TestBaseTab(unittest.TestCase):
    """ Test base_tab functionality. """

    def set_up_dummy_simulation(self,
                                simulation_path: Path,
                                planner_name: str,
                                scenario_type: str,
                                scenario_name: str,
                                ) -> None:
        """
        Set up dummy simulation data.
        :param simulation_path: Simulation path.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """

        json_file = Path(os.path.dirname(os.path.realpath(__file__))) / "json/test_simulation_tile.json"
        with open(json_file, "r") as f:
            simulation_data = json.load(f)

        # Save to a tmp folder
        save_path = simulation_path / planner_name / scenario_type / scenario_name
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / "1.json"
        with open(save_file, "w") as f:
            json.dump(simulation_data, f)

    def set_up_dummy_metric(self,
                            metric_path: Path,
                            planner_name: str,
                            scenario_type: str,
                            scenario_name: str
                            ) -> None:
        """
        Set up dummy metric results.
        :param metric_path: Metric path.
        :param planner_name: Planner name.
        :param scenario_type: Scenario type.
        :param scenario_name: Scenario name.
        """

        # Set up dummy metric statistics
        statistics = {MetricStatisticsType.MAX: Statistic(name="ego_max_acceleration", unit="meters_per_second_squared",
                                                          value=2.0),
                      MetricStatisticsType.MIN: Statistic(name="ego_min_acceleration", unit="meters_per_second_squared",
                                                          value=0.0),
                      MetricStatisticsType.P90: Statistic(name="ego_p90_acceleration", unit="meters_per_second_squared",
                                                          value=1.0),
                      }
        time_stamps = [0, 1, 2]
        accel = [0.0, 1.0, 2.0]
        time_series = TimeSeries(unit='meters_per_second_squared',
                                 time_stamps=list(time_stamps),
                                 values=list(accel))
        result = MetricStatistics(metric_computator='ego_acceleration',
                                  name="ego_acceleration_statistics",
                                  statistics=statistics,
                                  time_series=time_series,
                                  metric_category='Dynamic')

        # Set up dummy metric file
        key = MetricFileKey(metric_name='ego_acceleration',
                            scenario_name=scenario_name,
                            scenario_type=scenario_type,
                            planner_name=planner_name)
        metric_file = MetricFile(
            key=key,
            metric_statistics={'ego_acceleration': [result]}
        )

        # Set up a dummy metric engine and save the results to a metric file.
        metric_engine = MetricsEngine(
            main_save_path=metric_path,
            scenario_type=scenario_type,
            timestamp=0
        )

        metric_engine.save_metric_files(metric_files=[metric_file])

    def setUp(self) -> None:
        """ Set up a nuboard base tab. """

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.nuboard_file = NuBoardFile(main_path=self.tmp_dir.name,
                                        metric_folder='metrics',
                                        simulation_folder='simulations')
        doc = Document()
        planner_name = "SimplePlanner"
        scenario_type = "Test"
        scenario_name = "Dummy_scene"

        # Set up dummy metric files
        metric_path = Path(self.nuboard_file.main_path) / self.nuboard_file.metric_folder
        metric_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_metric(metric_path=metric_path,
                                 planner_name=planner_name,
                                 scenario_name=scenario_name,
                                 scenario_type=scenario_type
                                 )

        # Set up dummy simulation files
        simulation_path = Path(self.nuboard_file.main_path) / self.nuboard_file.simulation_folder
        simulation_path.mkdir(exist_ok=True, parents=True)
        self.set_up_dummy_simulation(simulation_path, planner_name=planner_name, scenario_type=scenario_type,
                                     scenario_name=scenario_name)

        self.base_tab = BaseTab(doc=doc,
                                file_paths=[self.nuboard_file])

    def test_panel(self) -> None:
        """ Test panel availability. """

        self.assertEqual(self.base_tab.panel, None)

    def test_load_metric_files(self) -> None:
        """ Test load_metric_files feature. """

        self.base_tab.load_metric_files(reset=True)
        self.assertEqual(len(self.base_tab.metric_scenario_keys), 1)

    def test_load_simulation_files(self) -> None:
        """ Test load_simulation_files feature. """

        self.base_tab.load_simulation_files(reset=True)
        self.assertEqual(len(self.base_tab.simulation_scenario_keys), 1)

    def test_file_paths_on_change(self) -> None:
        """ Test file_paths_on_change feature. """

        nuboard_files = [self.nuboard_file]
        self.assertRaises(NotImplementedError, self.base_tab.file_paths_on_change, nuboard_files)

    def tearDown(self) -> None:
        """ Remove all temporary folders and files. """

        self.tmp_dir.cleanup()
