import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas
from bokeh.document.document import Document
from bokeh.layouts import column, gridplot, layout
from bokeh.models import ColumnDataSource, Div, HoverTool, Select
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.base.simulation_tile import SimulationTile
from nuplan.planning.nuboard.style import PLOT_PALETTE, default_div_style, scenario_tab_style
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder

logger = logging.getLogger(__name__)


@dataclass
class ScenarioTimeSeriesData:
    """Time series data in the scenario tab."""

    experiment_index: int  # Experiment index to represent color.
    planner_name: str  # Planner name
    time_series_values: npt.NDArray[np.float64]  # A list of time series values
    time_series_timestamps: List[int]  # A list of time series timestamps
    time_series_unit: str  # Time series unit


class ScenarioTab(BaseTab):
    """Scenario tab in nuboard."""

    def __init__(
        self,
        doc: Document,
        experiment_file_data: ExperimentFileData,
        vehicle_parameters: VehicleParameters,
        scenario_builder: AbstractScenarioBuilder,
    ):
        """
        Scenario tab to render metric results about a scenario.
        :param doc: Bokeh HTML document.
        :param experiment_file_data: Experiment file data.
        :param vehicle_parameters: Vehicle parameters.
        :param scenario_builder: nuPlan scenario builder instance.
        """
        super().__init__(doc=doc, experiment_file_data=experiment_file_data)
        self.planner_checkbox_group.name = "scenario_planner_checkbox_group"
        self._scenario_builder = scenario_builder

        # UI.
        self._scalar_scenario_type_select = Select(
            name="scenario_scalar_scenario_type_select",
            css_classes=["scalar-scenario-type-select"],
        )
        self._scalar_scenario_type_select.on_change("value", self._scalar_scenario_type_select_on_change)
        self._scalar_log_name_select = Select(
            name="scenario_scalar_log_name_select",
            css_classes=["scalar-log-name-select"],
        )
        self._scalar_log_name_select.on_change("value", self._scalar_log_name_select_on_change)

        self._scalar_scenario_name_select = Select(
            name="scenario_scalar_scenario_name_select",
            css_classes=["scalar-scenario-name-select"],
        )
        self._scalar_scenario_name_select.on_change("value", self._scalar_scenario_name_select_on_change)
        self._loading_js = CustomJS(
            args={},
            code="""
            document.getElementById('scenario-loading').style.visibility = 'visible';
            document.getElementById('scenario-plot-section').style.visibility = 'hidden';
            cb_obj.tags = [window.outerWidth, window.outerHeight];
        """,
        )
        self._scalar_scenario_name_select.js_on_change("value", self._loading_js)
        self.planner_checkbox_group.js_on_change("active", self._loading_js)
        self._default_time_series_div = Div(
            text=""" <p> No time series results, please add more experiments or
                adjust the search filter.</p>""",
            css_classes=['scenario-time-series-default-div'],
            margin=default_div_style['margin'],
        )
        self._time_series_layout = column(
            self._default_time_series_div,
            css_classes=["scenario-time-series-layout"],
            name="time_series_layout",
        )

        self._default_simulation_div = Div(
            text=""" <p> No simulation data, please add more experiments or
                adjust the search filter.</p>""",
            css_classes=['scenario-simulation-default-div'],
            margin=default_div_style['margin'],
        )
        self._simulation_tile_layout = column(
            self._default_simulation_div,
            css_classes=["scenario-simulation-layout"],
            name="simulation_tile_layout",
        )
        self._end_loading_js = CustomJS(
            args={},
            code="""
            document.getElementById('scenario-loading').style.visibility = 'hidden';
            document.getElementById('scenario-plot-section').style.visibility = 'visible';
        """,
        )
        self._simulation_tile_layout.js_on_change("children", self._end_loading_js)
        self.simulation_tile = SimulationTile(
            map_factory=self._scenario_builder.get_map_factory(),
            doc=self._doc,
            vehicle_parameters=vehicle_parameters,
            experiment_file_data=experiment_file_data,
        )
        self._time_series_data: Dict[str, List[ScenarioTimeSeriesData]] = {}
        self._simulation_figure_data: List[Any] = []
        self._available_scenario_names: List[str] = []
        self._simulation_plots: Optional[column] = None
        self._init_selection()

    @property
    def scalar_scenario_type_select(self) -> Select:
        """Return scalar_scenario_type_select."""
        return self._scalar_scenario_type_select

    @property
    def scalar_log_name_select(self) -> Select:
        """Return scalar_log_name_select."""
        return self._scalar_log_name_select

    @property
    def scalar_scenario_name_select(self) -> Select:
        """Return scalar_scenario_name_select."""
        return self._scalar_scenario_name_select

    @property
    def time_series_layout(self) -> column:
        """Return time_series_layout."""
        return self._time_series_layout

    @property
    def simulation_tile_layout(self) -> column:
        """Return simulation_tile_layout."""
        return self._simulation_tile_layout

    def file_paths_on_change(
        self, experiment_file_data: ExperimentFileData, experiment_file_active_index: List[int]
    ) -> None:
        """
        Interface to update layout when file_paths is changed.
        :param experiment_file_data: Experiment file data.
        :param experiment_file_active_index: Active indexes for experiment files.
        """
        self._experiment_file_data = experiment_file_data
        self._experiment_file_active_index = experiment_file_active_index

        self.simulation_tile.init_simulations(figure_sizes=self.simulation_figure_sizes)
        self._init_selection()
        self._update_scenario_plot()

    def _click_planner_checkbox_group(self, attr: Any) -> None:
        """
        Click event handler for planner_checkbox_group.
        :param attr: Clicked attributes.
        """
        filtered_time_series_data: Dict[str, List[ScenarioTimeSeriesData]] = defaultdict(list)
        for key, time_series_data in self._time_series_data.items():
            for data in time_series_data:
                if data.planner_name not in self.enable_planner_names:
                    continue
                filtered_time_series_data[key].append(data)

        # Render time_series data
        time_series_plots = self._render_time_series(aggregated_time_series_data=filtered_time_series_data)
        self._time_series_layout.children[0] = layout(time_series_plots)

        # Render simulation
        filtered_simulation_figures = [
            data.plot for data in self._simulation_figure_data if data.planner_name in self.enable_planner_names
        ]
        if not filtered_simulation_figures:
            simulation_layouts = column(
                self._default_simulation_div,
                width=scenario_tab_style["default_div_width"],
                css_classes=["scenario-simulation-layout"],
                name="simulation_tile_layout",
            )
        else:
            simulation_layouts = gridplot(
                filtered_simulation_figures, ncols=self.get_simulation_plot_cols, toolbar_location=None
            )
        self._simulation_tile_layout.children[0] = layout(simulation_layouts)

    def _update_simulation_layouts(self) -> None:
        """Update simulation layouts."""
        self._simulation_tile_layout.children[0] = layout(self._simulation_plots)

    def _update_scenario_plot(self) -> None:
        """Update scenario plots when selection is made."""
        start_time = time.perf_counter()

        # Render time series.
        self._time_series_data = self._aggregate_time_series_data()
        time_series_plots = self._render_time_series(aggregated_time_series_data=self._time_series_data)
        self._time_series_layout.children[0] = layout(time_series_plots)

        # Render simulations.
        self._simulation_plots = self._render_simulations()

        # Make sure the simulation plot upgrades at the last
        self._doc.add_next_tick_callback(self._update_simulation_layouts)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.debug(f"Rending scenario plot takes {elapsed_time:.4f} seconds.")

    def _update_planner_names(self) -> None:
        """Update planner name options in the checkbox widget."""
        self.planner_checkbox_group.labels = []
        self.planner_checkbox_group.active = []
        selected_keys = [
            key
            for key in self.experiment_file_data.simulation_scenario_keys
            if key.scenario_type == self._scalar_scenario_type_select.value
            and key.scenario_name == self._scalar_scenario_name_select.value
        ]
        sorted_planner_names = sorted(list({key.planner_name for key in selected_keys}))
        self.planner_checkbox_group.labels = sorted_planner_names
        self.planner_checkbox_group.active = [index for index in range(len(sorted_planner_names))]

    def _scalar_scenario_type_select_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in scalar scenario type.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        if new == "":
            return

        available_log_names = self.load_log_name(scenario_type=self._scalar_scenario_type_select.value)
        self._scalar_log_name_select.options = [""] + available_log_names
        self._scalar_log_name_select.value = ""

    def _scalar_log_name_select_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in scalar log name.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        if new == "":
            return

        available_scenario_names = self.load_scenario_names(
            scenario_type=self._scalar_scenario_type_select.value, log_name=self._scalar_log_name_select.value
        )
        self._scalar_scenario_name_select.options = [""] + available_scenario_names
        self._scalar_scenario_name_select.value = ""

    def _scalar_scenario_name_select_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in scalar scenario name.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        if self._scalar_scenario_name_select.tags:
            self.window_width = self._scalar_scenario_name_select.tags[0]
            self.window_height = self._scalar_scenario_name_select.tags[1]
        self._update_planner_names()
        self._update_scenario_plot()

    def _init_selection(self) -> None:
        """Init histogram and scalar selection options."""
        self._scalar_scenario_type_select.value = ""
        self._scalar_scenario_type_select.options = []
        self._scalar_scenario_name_select.value = ""
        self._scalar_scenario_name_select.options = []
        self._available_scenario_names = []

        if len(self._scalar_scenario_type_select.options) == 0:
            self._scalar_scenario_type_select.options = [""] + self.experiment_file_data.available_scenario_types

        if len(self._scalar_scenario_type_select.options) > 0:
            self._scalar_scenario_type_select.value = self._scalar_scenario_type_select.options[0]
        self._update_planner_names()

    def _render_time_series_plot(self, title: str, y_axis_label: str) -> figure:
        """
        Render a time series plot.
        :param title: Plot title.
        :param y_axis_label: Y axis label.
        :return A time series plot.
        """
        time_series_figure = figure(
            background_fill_color=PLOT_PALETTE["background_white"],
            title=title,
            css_classes=["time-series-figure"],
            margin=scenario_tab_style["time_series_figure_margins"],
            width=self.plot_sizes[0],
            height=self.plot_sizes[1],
            active_scroll="wheel_zoom",
            output_backend="webgl",
        )
        hover = HoverTool(
            tooltips=[("Frame", "@x"), ("Time_us", "@time_us"), ("Value", "@y{0.0000}"), ("Planner", "$name")]
        )
        time_series_figure.add_tools(hover)

        time_series_figure.title.text_font_size = scenario_tab_style["time_series_figure_title_text_font_size"]
        time_series_figure.xaxis.axis_label_text_font_size = scenario_tab_style[
            "time_series_figure_xaxis_axis_label_text_font_size"
        ]
        time_series_figure.xaxis.major_label_text_font_size = scenario_tab_style[
            "time_series_figure_xaxis_major_label_text_font_size"
        ]
        time_series_figure.yaxis.axis_label_text_font_size = scenario_tab_style[
            "time_series_figure_yaxis_axis_label_text_font_size"
        ]
        time_series_figure.yaxis.major_label_text_font_size = scenario_tab_style[
            "time_series_figure_yaxis_major_label_text_font_size"
        ]
        time_series_figure.toolbar.logo = None

        # Rotate the x_axis label with 45 (180/4) degrees.
        time_series_figure.xaxis.major_label_orientation = np.pi / 4

        time_series_figure.yaxis.axis_label = y_axis_label
        time_series_figure.xaxis.axis_label = scenario_tab_style["time_series_figure_xaxis_axis_label"]

        return time_series_figure

    def _render_time_series_layout(self, time_series_figures: Dict[str, figure]) -> column:
        """
        Render time series layout.
        :param time_series_figures: A dictionary of time series plots.
        :return: A list of lists of figures (a list per row).
        """
        figures: List[figure] = []
        for metric_statistic_name, time_series_figure in time_series_figures.items():
            time_series_figure.legend.label_text_font_size = scenario_tab_style["plot_legend_label_text_font_size"]
            figures.append(time_series_figure)

        grid_plot = gridplot(figures, ncols=self.get_plot_cols, toolbar_location="left")
        time_series_layout = column(grid_plot)

        return time_series_layout

    def _aggregate_time_series_data(self) -> Dict[str, List[ScenarioTimeSeriesData]]:
        """
        Aggregate time series data.
        :return A dict of metric statistic names and their data.
        """
        aggregated_time_series_data: Dict[str, List[ScenarioTimeSeriesData]] = {}
        scenario_types = (
            tuple([self._scalar_scenario_type_select.value]) if self._scalar_scenario_type_select.value else None
        )
        log_names = tuple([self._scalar_log_name_select.value]) if self._scalar_log_name_select.value else None

        if not len(self._scalar_scenario_name_select.value):
            return aggregated_time_series_data

        for index, metric_statistics_dataframes in enumerate(self.experiment_file_data.metric_statistics_dataframes):
            if index not in self._experiment_file_active_index:
                continue

            for metric_statistics_dataframe in metric_statistics_dataframes:
                planner_names = metric_statistics_dataframe.planner_names
                if metric_statistics_dataframe.metric_statistic_name not in aggregated_time_series_data:
                    aggregated_time_series_data[metric_statistics_dataframe.metric_statistic_name] = []
                for planner_name in planner_names:
                    data_frame = metric_statistics_dataframe.query_scenarios(
                        scenario_names=tuple([str(self._scalar_scenario_name_select.value)]),
                        scenario_types=scenario_types,
                        planner_names=tuple([planner_name]),
                        log_names=log_names,
                    )
                    if not len(data_frame):
                        continue

                    time_series_headers = metric_statistics_dataframe.time_series_headers
                    time_series: pandas.DataFrame = data_frame[time_series_headers]
                    if time_series[time_series_headers[0]].iloc[0] is None:
                        continue

                    time_series_values: npt.NDArray[np.float64] = np.round(
                        np.asarray(
                            list(
                                chain.from_iterable(time_series[metric_statistics_dataframe.time_series_values_column])
                            )
                        ),
                        4,
                    )

                    time_series_timestamps = list(
                        chain.from_iterable(time_series[metric_statistics_dataframe.time_series_timestamp_column])
                    )
                    time_series_unit = time_series[metric_statistics_dataframe.time_series_unit_column].iloc[0]

                    scenario_time_series_data = ScenarioTimeSeriesData(
                        experiment_index=index,
                        planner_name=planner_name,
                        time_series_values=time_series_values,
                        time_series_timestamps=time_series_timestamps,
                        time_series_unit=time_series_unit,
                    )

                    aggregated_time_series_data[metric_statistics_dataframe.metric_statistic_name].append(
                        scenario_time_series_data
                    )

        return aggregated_time_series_data

    def _render_time_series(self, aggregated_time_series_data: Dict[str, List[ScenarioTimeSeriesData]]) -> column:
        """
        Render time series plots.
        :param aggregated_time_series_data: Aggregated scenario time series data.
        :return A column.
        """
        time_series_figures: Dict[str, figure] = {}
        for metric_statistic_name, scenario_time_series_data in aggregated_time_series_data.items():
            for data in scenario_time_series_data:
                if not len(data.time_series_values):
                    continue

                if metric_statistic_name not in time_series_figures:
                    time_series_figures[metric_statistic_name] = self._render_time_series_plot(
                        title=metric_statistic_name, y_axis_label=data.time_series_unit
                    )
                planner_name = data.planner_name + f" ({self.get_file_path_last_name(data.experiment_index)})"
                color = self.experiment_file_data.file_path_colors[data.experiment_index][data.planner_name]
                time_series_figure = time_series_figures[metric_statistic_name]
                data_source = ColumnDataSource(
                    dict(
                        x=list(range(len(data.time_series_values))),
                        y=data.time_series_values,
                        time_us=data.time_series_timestamps,
                    )
                )
                time_series_figure.line(
                    x="x", y="y", name=planner_name, color=color, legend_label=planner_name, source=data_source
                )

        if not time_series_figures:
            time_series_column = column(
                self._default_time_series_div,
                width=scenario_tab_style["default_div_width"],
                css_classes=["scenario-simulation-layout"],
                name="simulation_tile_layout",
            )
        else:
            time_series_column = self._render_time_series_layout(time_series_figures=time_series_figures)
        return time_series_column

    def _render_simulations(self) -> column:
        """
        Render simulation plot.
        :return: A list of Bokeh columns or rows.
        """
        selected_keys = [
            key
            for key in self.experiment_file_data.simulation_scenario_keys
            if key.scenario_type == self._scalar_scenario_type_select.value
            and key.log_name == self._scalar_log_name_select.value
            and key.scenario_name == self._scalar_scenario_name_select.value
            and key.nuboard_file_index in self._experiment_file_active_index
        ]
        if not selected_keys:
            simulation_layouts = column(
                self._default_simulation_div,
                width=800,
                css_classes=["scenario-simulation-layout"],
                name="simulation_tile_layout",
            )
        else:
            self._simulation_figure_data = self.simulation_tile.render_simulation_tiles(
                selected_scenario_keys=selected_keys, figure_sizes=self.simulation_figure_sizes
            )
            simulation_figures = [data.plot for data in self._simulation_figure_data]
            simulation_layouts = gridplot(
                simulation_figures, ncols=self.get_simulation_plot_cols, toolbar_location=None
            )

        return simulation_layouts
