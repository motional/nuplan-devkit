import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from bokeh.document.document import Document
from bokeh.layouts import column, gridplot, row
from bokeh.models import Div, Panel, Select
from bokeh.plotting import figure
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.base.simulation_tile import SimulationTile
from nuplan.planning.nuboard.style import MOTIONAL_PALETTE, PLOT_PALETTE, scenario_tab_style
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder

logger = logging.getLogger(__name__)


class ScenarioTab(BaseTab):

    def __init__(self,
                 doc: Document,
                 vehicle_parameters: VehicleParameters,
                 scenario_builder: AbstractScenarioBuilder,
                 file_paths: List[NuBoardFile]):
        """
        Metric board to render metrics.
        :param doc: Bokeh html document.
        :param vehicle_parameters: Vehicle parameters.
        :param scenario_builder: nuPlan scenario builder instance.
        :param file_paths: Path to a metric pickle file.
        """

        super().__init__(doc=doc,
                         file_paths=file_paths)
        self._scenario_builder = scenario_builder

        # UI.
        self._scalar_scenario_type_select = Select(title="Scenario type:",
                                                   margin=self.search_criteria_margin)
        self._scalar_scenario_type_select.on_change("value", self._scalar_scenario_type_select_on_change)

        self._scalar_scenario_name_select = Select(title="Scenario:",
                                                   margin=self.search_criteria_margin)
        self._scalar_scenario_name_select.on_change("value", self._scalar_scenario_name_select_on_change)

        search_criteria = column(self.search_criteria_title,
                                 self._scalar_scenario_type_select, self._scalar_scenario_name_select,
                                 css_classes=["search-criteria-panel"], height=self.search_criteria_height)

        self._time_series_title = Div(text="""<h3 style='margin-left: 18px; margin-top: 20px; width: 200px;'>
        Time series</h3>""")
        self._simulation_title = Div(text="""<h3 style='margin-left: 18px; margin-top: 20px; width: 200px;'>
        Simulation</h3>""")
        time_series_frame = column(self._simulation_title,
                                   Div(),
                                   self._time_series_title,
                                   Div(),
                                   css_classes=["scenario-time-series-panel"],
                                   sizing_mode="scale_height",
                                   height=self.plot_frame_sizes[1])

        self._plot_layout = row([search_criteria, time_series_frame], css_classes=["scenario-tab"])
        self._panel = Panel(title="Scenarios", child=self._plot_layout)

        self.simulation_tile = SimulationTile(scenario_builder=self._scenario_builder,
                                              doc=self._doc,
                                              vehicle_parameters=vehicle_parameters)

        self._init_selection()

    def file_paths_on_change(self, file_paths: List[NuBoardFile]) -> None:
        """ Update if file_path is changed. """

        self._file_paths = file_paths

        self._scalar_scenario_type_select.value = ''
        self._scalar_scenario_type_select.options = []
        self._scalar_scenario_name_select.value = ''
        self._scalar_scenario_name_select.options = []
        self._update_scenario_plot()

        self.load_metric_files(reset=True)
        self.load_simulation_files(reset=True)
        self._init_selection()

    def _update_scenario_plot(self) -> None:
        """ Update scenario plots when selection is made. """

        start_time = time.perf_counter()

        # Render time series.
        time_series_plots = self._render_time_series()

        # Render simulations.
        simulation_layouts = self._render_simulations()

        col = column([self._simulation_title] +
                     simulation_layouts +
                     [self._time_series_title] +
                     time_series_plots,
                     css_classes=["scenario-time-series-panel"],
                     sizing_mode="scale_height",
                     height=self.plot_frame_sizes[1])

        self._plot_layout.children[1] = col

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.debug(f"Rending scenario plot takes {elapsed_time:.4f} seconds.")

    def _scalar_scenario_type_select_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in scalar scenario type.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """

        if new == '':
            return

        # Update metric options.
        scenario_names = list({scenario_key.scenario_name for scenario_key in self.simulation_scenario_keys
                               if scenario_key.scenario_type == new})
        scenario_names = sorted(scenario_names, reverse=False)

        self._scalar_scenario_name_select.options = []
        self._scalar_scenario_name_select.options = [''] + scenario_names
        self._scalar_scenario_name_select.value = ''

    def _scalar_scenario_name_select_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in scalar scenario name.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """

        # Do nothing
        if new == '':
            return

        self._update_scenario_plot()

    def _init_selection(self) -> None:
        """ Init histogram and scalar selection options. """

        scenario_keys = self.simulation_scenario_keys

        # Scenario types.
        scenario_types: List[str] = list({key.scenario_type for key in scenario_keys})
        scenario_types = sorted(list(scenario_types), reverse=False)
        if len(self._scalar_scenario_type_select.options) == 0:
            self._scalar_scenario_type_select.options = scenario_types

        if len(self._scalar_scenario_type_select.options) > 0:
            self._scalar_scenario_type_select.value = self._scalar_scenario_type_select.options[0]

    def _render_time_series_plot(self, title: str, y_axis_label: str) -> figure:
        """
        Render a time series plot.
        :param title: Plot title.
        :param y_axis_label: Y axis label.
        :return A time series plot.
        """

        time_series_figure = figure(background_fill_color=PLOT_PALETTE['background_white'],
                                    title=title,
                                    css_classes=['time-series-figure'],
                                    margin=scenario_tab_style['time_series_figure_margins'],
                                    height=self.plot_sizes[1])

        time_series_figure.title.text_font_size = scenario_tab_style['time_series_figure_title_text_font_size']
        time_series_figure.xaxis.axis_label_text_font_size = \
            scenario_tab_style['time_series_figure_xaxis_axis_label_text_font_size']
        time_series_figure.xaxis.major_label_text_font_size = \
            scenario_tab_style['time_series_figure_xaxis_major_label_text_font_size']
        time_series_figure.yaxis.axis_label_text_font_size = \
            scenario_tab_style['time_series_figure_yaxis_axis_label_text_font_size']
        time_series_figure.yaxis.major_label_text_font_size = \
            scenario_tab_style['time_series_figure_yaxis_major_label_text_font_size']
        time_series_figure.toolbar.logo = None

        # Rotate the x_axis label with 45 (180/4) degrees.
        time_series_figure.xaxis.major_label_orientation = np.pi / 4

        time_series_figure.yaxis.axis_label = y_axis_label
        time_series_figure.xaxis.axis_label = scenario_tab_style['time_series_figure_xaxis_axis_label']

        return time_series_figure

    def _render_time_series_layout(self, time_series: Dict[str, Dict[str, figure]]) -> List[column]:
        """
        Render time series layout.
        :param time_series: A dictionary of time series plots.
        :return: A list of lists of figures (a list per row).
        """

        figures: List[figure] = []
        for metric_name, metric_result_figures in time_series.items():
            for metric_result_name, time_series_figure in metric_result_figures.items():
                plot = time_series_figure[0]
                plot.legend.label_text_font_size = scenario_tab_style['plot_legend_label_text_font_size']
                figures.append(plot)
        grid_plot = gridplot(figures, ncols=self.plot_cols)
        grid_layout = [column(grid_plot)]

        return grid_layout

    def _render_time_series(self) -> List[column]:
        """
        Render time series plots.
        :return A list of columns.
        """

        selected_keys = [key for key in self.metric_scenario_keys
                         if key.scenario_type == self._scalar_scenario_type_select.value and
                         key.scenario_name == self._scalar_scenario_name_select.value]

        if len(selected_keys) == 0:
            return []

        metric_files = [self._read_metric_file(key.file) for key in selected_keys]
        color_keys = list(MOTIONAL_PALETTE.keys())
        figures: Dict[str, Dict[str, Tuple[figure, int]]] = {}

        for metric_file in metric_files:
            key = metric_file.key
            metric_name = key.metric_name
            planner_name = key.planner_name

            for statistic_group_name, metric_statistics in metric_file.metric_statistics.items():
                for metric_statistic in metric_statistics:
                    time_series = metric_statistic.time_series
                    if time_series is None:
                        continue

                    values = np.asarray(time_series.values)
                    values = values[np.isfinite(values)]

                    if not len(values):
                        continue

                    if metric_name not in figures:
                        figures[metric_name] = {}

                    if statistic_group_name not in figures[metric_name]:
                        time_series_figure = self._render_time_series_plot(title=statistic_group_name,
                                                                           y_axis_label=time_series.unit)
                        color_index = 0
                        figures[metric_name][statistic_group_name] = (time_series_figure, color_index)
                    else:
                        time_series_figure, color_index = figures[metric_name][statistic_group_name]
                        color_index += 1
                        figures[metric_name][statistic_group_name] = (time_series_figure, color_index)

                    color = MOTIONAL_PALETTE[color_keys[color_index]]
                    time_series_figure.line(x=list(range(len(values))),
                                            y=values,
                                            color=color,
                                            legend_label=planner_name)
        layout = self._render_time_series_layout(time_series=figures)
        return layout

    def _render_simulations(self) -> List[Any]:
        """
        Render simulation plot.
        :return: A list of Bokeh columns or rows.
        """

        selected_keys = [key for key in self.simulation_scenario_keys
                         if key.scenario_type == self._scalar_scenario_type_select.value and
                         key.scenario_name == self._scalar_scenario_name_select.value]

        if len(selected_keys) == 0:
            return []

        layouts = self.simulation_tile.render_simulation_tiles(selected_scenario_keys=selected_keys)
        return layouts  # type: ignore
