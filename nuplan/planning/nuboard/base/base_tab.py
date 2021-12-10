import logging
import pathlib
import pickle
from collections import defaultdict
from typing import Any, Dict, List

from bokeh.document.document import Document
from bokeh.models import Div, MultiChoice, Panel
from nuplan.planning.metrics.metric_file import MetricFile
from nuplan.planning.nuboard.base.data_class import MetricScenarioKey, NuBoardFile, SimulationScenarioKey
from nuplan.planning.nuboard.style import base_tab_style

logger = logging.getLogger(__name__)


class BaseTab:

    def __init__(self,
                 doc: Document,
                 file_paths: List[NuBoardFile]):
        """
        Base tabs for common properties.
        Metric board to render metrics.
        :doc: A bokeh HTML document.
        :param file_paths: Path to a metric pickle file.
        """

        self._doc = doc
        self._file_paths = file_paths
        if file_paths is None:
            raise ValueError("file_path must be given.")

        self._metric_scenario_keys: List[MetricScenarioKey] = []
        self._simulation_scenario_keys: List[SimulationScenarioKey] = []
        self._simulation_files: Dict[str, Any] = defaultdict(set)
        self._metric_files: Dict[str, Any] = defaultdict()
        self.load_metric_files(reset=True)
        self.load_simulation_files(reset=True)

        # UI.
        self.search_criteria_title = Div(text="""<h3 style='margin-left: 18px;
        margin-top: 20px;'>Search Criteria</h3>""", width=base_tab_style['search_criteria_title_width'])

        self.fig_frame_minimum_height = base_tab_style['fig_frame_minimum_height']
        self.search_criteria_height = base_tab_style['search_criteria_height']
        self.plot_sizes = base_tab_style['plot_sizes']
        self.plot_frame_sizes = base_tab_style['plot_frame_sizes']
        self.plot_cols = 2

        # Top, right, bottom, left
        self.search_criteria_margin = base_tab_style['search_criteria_margin']
        self._panel = None

    def file_paths_on_change(self, file_paths: List[NuBoardFile]) -> None:
        """
        Interface to update layout when file_paths is changed.
        :param file_paths: A list of new file paths.
        """

        raise NotImplementedError

    @property
    def panel(self) -> Panel:
        """
        Access a panel tab.
        :return A panel tab.
        """

        return self._panel

    @property
    def simulation_scenario_keys(self) -> List[SimulationScenarioKey]:
        """
        Return a list of scenario keys.
        :return: A list of scenario metric keys.
        """

        if len(self._simulation_scenario_keys) == 0:
            keys = [{'key': key.split("/"), 'files': list(files)} for key, files in self._simulation_files.items()]
            self._simulation_scenario_keys = [SimulationScenarioKey(
                planner_name=key['key'][1], scenario_type=key['key'][2], scenario_name=key['key'][3],
                files=key['files']) for key in keys]

        return self._simulation_scenario_keys

    @property
    def metric_scenario_keys(self) -> List[MetricScenarioKey]:
        """
        Return a list of scenario keys.
        :return: A list of scenario metric keys.
        """

        if len(self._metric_scenario_keys) == 0:
            keys = [{'key': key.split("/"), 'file': file} for key, file in self._metric_files.items()]
            self._metric_scenario_keys = [MetricScenarioKey(planner_name=key['key'][1], scenario_type=key['key'][2],
                                                            metric_result_name=key['key'][3],
                                                            scenario_name=key['key'][4],
                                                            file=key['file']) for key in keys]

        return self._metric_scenario_keys

    def load_metric_files(self, reset: bool = False) -> None:
        """
        Load metric files.
        Folder hierarchy: planner_name -> scenario_type -> metric result name -> scenario_name.pkl
        :param reset: Reset all files.
        """

        if reset:
            self._metric_files = defaultdict()
            self._metric_scenario_keys = []

        for file_path in self._file_paths:
            base_path = pathlib.Path(file_path.main_path)
            metric_path = base_path / file_path.metric_folder
            planner_name_paths = metric_path.iterdir()
            for planner_name_path in planner_name_paths:
                scenario_type_paths = planner_name_path.iterdir()
                for scenario_type_path in scenario_type_paths:
                    metric_result_name_paths = scenario_type_path.iterdir()
                    for metric_result_name_path in metric_result_name_paths:
                        scenario_name_metric_files = metric_result_name_path.iterdir()
                        for scenario_name_metric_file in scenario_name_metric_files:
                            scenario_key = f"{base_path.name}/{planner_name_path.name}/{scenario_type_path.name}/" \
                                           f"{metric_result_name_path.name}/{scenario_name_metric_file.stem}"
                            self._metric_files[scenario_key] = scenario_name_metric_file

    def load_simulation_files(self, reset: bool = False) -> None:
        """
        Load simulation files.
        Folder hierarchy: planner_name -> scenario_type -> scenario_names -> iteration.pkl.
        :param reset: Reset all files.
        """

        if reset:
            self._simulation_files = defaultdict(set)
            self._simulation_scenario_keys = []

        for file_path in self._file_paths:
            base_path = pathlib.Path(file_path.main_path)
            simulation_path = base_path / file_path.simulation_folder
            planner_name_paths = simulation_path.iterdir()
            for planner_name_path in planner_name_paths:
                planner_name = planner_name_path.name
                scenario_type_paths = planner_name_path.iterdir()
                for scenario_type_path in scenario_type_paths:
                    scenario_name_paths = scenario_type_path.iterdir()
                    scenario_type = scenario_type_path.name
                    for scenario_name_path in scenario_name_paths:
                        scenario_name = scenario_name_path.name
                        scenario_key = f"{base_path.name}/{planner_name}/{scenario_type}/{scenario_name}"
                        files = scenario_name_path.iterdir()
                        for file in files:
                            self._simulation_files[scenario_key].add(file)

    @staticmethod
    def _read_metric_file(metric_file_path: pathlib.Path) -> MetricFile:
        """
        Read a metric result pkl file.
        :param metric_file_path: Metric file path.
        :return A metric storage result.
        """

        with open(metric_file_path, "rb") as f:
            data = pickle.load(f)
            metric_file = MetricFile.deserialize(data=data)

        return metric_file

    def _init_multi_search_criteria_selection(self,
                                              scenario_type_multi_choice: MultiChoice,
                                              metric_name_multi_choice: MultiChoice) -> None:
        """
        Init histogram and scenario selection options.
        :param scenario_type_multi_choice: Scenario type multi choice.
        :param metric_name_multi_choice: Metric type multi choice.
        """

        scenario_keys = self.metric_scenario_keys

        # Scenario types.
        scenario_types = list({key.scenario_type for key in scenario_keys})
        scenario_types = sorted(scenario_types, reverse=False)
        if len(scenario_type_multi_choice.options) == 0:
            scenario_type_multi_choice.options = scenario_types

        # Metrics results
        metric_results = list({key.metric_result_name for key in scenario_keys})
        metric_results = sorted(metric_results, reverse=False)
        if len(metric_name_multi_choice.options) == 0:
            metric_name_multi_choice.options = metric_results
