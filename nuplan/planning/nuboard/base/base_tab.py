import logging
import pathlib
from typing import Any, List, Optional

from bokeh.document.document import Document
from bokeh.models import CheckboxGroup, MultiChoice
from bokeh.plotting.figure import Figure

from nuplan.planning.nuboard.base.data_class import SelectedMetricStatisticDataFrame, SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.style import base_tab_style, simulation_tile_style

logger = logging.getLogger(__name__)


class BaseTab:
    """Base tab for other tabs."""

    def __init__(self, doc: Document, experiment_file_data: ExperimentFileData):
        """
        Base tabs for common properties.
        Metric board to render metrics.
        :doc: A bokeh HTML document.
        :param experiment_file_data: Experiment file data.
        """
        self._doc = doc
        self._experiment_file_data: ExperimentFileData = experiment_file_data

        self._simulation_scenario_keys: List[SimulationScenarioKey] = []
        self._experiment_file_active_index: List[int] = []

        # UI
        self.scatter_signs = [
            'circle',
            'diamond',
            'plus',
            'square',
            'triangle',
            'inverted_triangle',
            'star',
            'asterisk',
            'dot_circle',
            'diamond_cross',
        ]
        self.search_criteria_selection_size = base_tab_style["search_criteria_sizes"]
        self.plot_sizes = base_tab_style["plot_sizes"]
        self.simulation_figure_sizes = simulation_tile_style["figure_sizes"]
        self.plot_frame_sizes = base_tab_style["plot_frame_sizes"]
        self.window_width = 0
        self.window_height = 0

        self.planner_checkbox_group = CheckboxGroup(
            labels=[], active=[], inline=True, css_classes=["planner-checkbox-group"], sizing_mode="scale_both"
        )
        self.planner_checkbox_group.on_click(self._click_planner_checkbox_group)

    def file_paths_on_change(
        self, experiment_file_data: ExperimentFileData, experiment_file_active_index: List[int]
    ) -> None:
        """
        Interface to update layout when file_paths is changed.
        :param experiment_file_data: Experiment file data.
        :param experiment_file_active_index: Active indexes for experiment files.
        """
        raise NotImplementedError

    def _click_planner_checkbox_group(self, attr: Any) -> None:
        """
        Click event handler for planner_checkbox_group.
        :param attr: Clicked attributes.
        """
        raise NotImplementedError

    @property
    def experiment_file_data(self) -> ExperimentFileData:
        """Return experiment file data."""
        return self._experiment_file_data

    @experiment_file_data.setter
    def experiment_file_data(self, experiment_file_data: ExperimentFileData) -> None:
        """
        Update experiment file data.
        :param experiment_file_data: New experiment file data.
        """
        self._experiment_file_data = experiment_file_data

    @property
    def enable_planner_names(self) -> List[str]:
        """Return a list of enable planner names."""
        enable_planner_names = [
            self.planner_checkbox_group.labels[index] for index in self.planner_checkbox_group.active
        ]
        return enable_planner_names

    def get_plot_cols(
        self, plot_width: int, default_col_width: int = 1024, offset_width: int = 0, default_ncols: int = 0
    ) -> int:
        """
        Return number of columns for a grid plot.
        :param plot_width: Plot width.
        :param default_col_width: The number of columns would be 1 if window width is lower than this value.
        :param offset_width: Additional offset width.
        :param default_ncols: Default number of columns.
        :return: Get a number of columns for a grid plot.
        """
        if default_ncols and not self.window_width:
            return default_ncols

        window_width = self.window_width - offset_width
        if window_width <= default_col_width:
            return 1
        col_num = 1 + round((window_width - default_col_width) / plot_width)
        return col_num

    def get_scatter_sign(self, index: int) -> str:
        """
        Get scatter index sign based on the index.
        :param index: Index for the scatter sign.
        :return A scatter sign name.
        """
        index = index % len(self.scatter_signs)  # Repeat if out of the available scatter signs
        return self.scatter_signs[index]

    @staticmethod
    def get_scatter_render_func(scatter_sign: str, scatter_figure: Figure) -> Any:
        """
        Render a scatter plot.
        :param scatter_sign: Scatter sign.
        :param scatter_figure: Scatter figure.
        :return A scatter render function.
        """
        if scatter_sign == 'circle':
            renderer = scatter_figure.circle
        elif scatter_sign == 'diamond':
            renderer = scatter_figure.diamond
        elif scatter_sign == 'plus':
            renderer = scatter_figure.plus
        elif scatter_sign == 'square':
            renderer = scatter_figure.square
        elif scatter_sign == 'triangle':
            renderer = scatter_figure.triangle
        elif scatter_sign == 'inverted_triangle':
            renderer = scatter_figure.inverted_triangle
        elif scatter_sign == 'star':
            renderer = scatter_figure.star
        elif scatter_sign == 'asterisk':
            renderer = scatter_figure.asterisk
        elif scatter_sign == 'diamond_cross':
            renderer = scatter_figure.diamond_cross
        else:
            raise NotImplementedError(f"{scatter_sign} is not a valid option for scatter plots!")

        return renderer

    def get_file_path_last_name(self, index: int) -> str:
        """
        Get last name of a file path.
        :param index: Index for the file path.
        :return: A file path string name.
        """
        file_path = self._experiment_file_data.file_paths[index]
        default_experiment_file_path_stem = pathlib.Path(file_path.metric_main_path)
        if file_path.current_path is None:
            return str(default_experiment_file_path_stem.name)

        metric_path = pathlib.Path(file_path.current_path, file_path.metric_folder)
        if metric_path.exists():
            experiment_file_path_stem = file_path.current_path
        else:
            experiment_file_path_stem = default_experiment_file_path_stem
        return str(experiment_file_path_stem.name)

    def load_log_name(self, scenario_type: str) -> List[str]:
        """
        Load a list of log names based on the scenario type.
        :param scenario_type: A selected scenario type.
        :return a list of log names.
        """
        log_names = self._experiment_file_data.available_scenarios.get(scenario_type, [])

        # Remove duplicates
        sorted_log_names: List[str] = sorted(list(set(log_names)), reverse=False)

        return sorted_log_names

    def load_scenario_names(self, scenario_type: str, log_name: str) -> List[str]:
        """
        Load a list of scenario names based on the log name.
        :param scenario_type: A selected scenario type.
        :param log_name: A selected log name.
        :return a list of scenario names.
        """
        log_dict = self._experiment_file_data.available_scenarios.get(scenario_type, [])
        if not log_dict:
            return []

        scenario_names = log_dict.get(log_name, [])

        # Remove duplicates
        sorted_scenario_names: List[str] = sorted(list(set(scenario_names)), reverse=False)

        return sorted_scenario_names

    def _init_multi_search_criteria_selection(
        self, scenario_type_multi_choice: MultiChoice, metric_name_multi_choice: MultiChoice
    ) -> None:
        """
        Init histogram and scenario selection options.
        :param scenario_type_multi_choice: Scenario type multi choice.
        :param metric_name_multi_choice: Metric type multi choice.
        """
        # Scenario types.
        scenario_type_multi_choice.options = ['all'] + sorted(self.experiment_file_data.available_scenario_types)

        # Metrics results
        metric_name_multi_choice.options = sorted(self.experiment_file_data.available_metric_statistics_names)

    def search_metric_statistics_dataframe(
        self, scenario_types: Optional[List[str]] = None, metric_choices: Optional[List[str]] = None
    ) -> List[SelectedMetricStatisticDataFrame]:
        """
        Search metric statistics dataframe based on scenario types and metric choices.
        :param scenario_types: A list of scenario types.
        :param metric_choices: A list of metric choices.
        :return: A list of selected metric statistic dataframe.
        """
        data: List[SelectedMetricStatisticDataFrame] = []
        if not scenario_types and not metric_choices:
            return data

        # Loop through all metric statistics in the dataframe
        for index, metric_statistics_dataframes in enumerate(self.experiment_file_data.metric_statistics_dataframes):
            for metric_statistics_dataframe in metric_statistics_dataframes:

                # Only run if it matches with the searched values or types
                if metric_choices and metric_statistics_dataframe.metric_statistic_name not in metric_choices:
                    continue

                data.append(
                    SelectedMetricStatisticDataFrame(dataframe_index=index, dataframe=metric_statistics_dataframe)
                )

        return data
