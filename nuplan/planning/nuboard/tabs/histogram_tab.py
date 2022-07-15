import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from bokeh.document.document import Document
from bokeh.layouts import column, gridplot, layout
from bokeh.models import ColumnDataSource, Div, MultiChoice, Spinner, glyph
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure

from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.style import (
    PLOT_PALETTE,
    default_div_style,
    default_multi_choice_style,
    default_spinner_style,
    histogram_tab_style,
)

logger = logging.getLogger(__name__)


@dataclass
class HistogramStatistics:
    """Histogram statistics data."""

    values: npt.NDArray[np.float64]  # An array of values
    unit: str  # Unit
    scenarios: List[str]  # Scenario names


@dataclass
class HistogramData:
    """Histogram data."""

    experiment_index: int  # Experiment index to represent color
    planner_name: str  # Planner name
    statistics: Dict[str, HistogramStatistics]  # Aggregated statistic data


@dataclass
class HistogramFigureData:
    """Histogram figure data."""

    figure_plot: figure  # Histogram statistic figure
    frequency_array: Optional[npt.NDArray[np.int64]] = None


@dataclass
class HistogramEdgeData:
    """Histogram edge data."""

    unit: str  # Unit
    values: npt.NDArray[np.float64]  # An array of values


# Type for histogram aggregated data: {metric name: A list of histogram aggregated data}
histogram_data_type = Dict[str, List[HistogramData]]

# Type for histogram figure data type: {metric name: {metric statistics name: histogram figure data}}
histogram_figures_type = Dict[str, Dict[str, HistogramFigureData]]

# Type for histogram edge data type: {metric name: {metric statistic name: histogram figure data}}
histogram_edges_data_type = Dict[str, Dict[str, Optional[npt.NDArray[np.float64]]]]


class HistogramTab(BaseTab):
    """Histogram tab in nuBoard."""

    def __init__(
        self, doc: Document, experiment_file_data: ExperimentFileData, bins: int = 20, max_scenario_names: int = 20
    ):
        """
        Histogram for metric results about simulation.
        :param doc: Bokeh html document.
        :param experiment_file_data: Experiment file data.
        :param bins: Default number of bins in histograms.
        :param max_scenario_names: Show the maximum list of scenario names in each bin, 0 or None to disable
        """
        super().__init__(doc=doc, experiment_file_data=experiment_file_data)
        self._bins = bins
        self._max_scenario_names = max_scenario_names

        # UI.
        # Planner selection
        self.planner_checkbox_group.name = "histogram_planner_checkbox_group"

        # Scenario type multi choices
        self._scenario_type_multi_choice = MultiChoice(
            css_classes=["scenario-type-multi-choice"],
            name="histogram_scenario_type_multi_choice",
            option_limit=default_multi_choice_style['option_limit'],
        )
        self._scenario_type_multi_choice.on_change("value", self._scenario_type_multi_choice_on_change)
        self._loading_js = CustomJS(
            args={},
            code="""
                    document.getElementById('histogram-loading').style.visibility = 'visible';
                    document.getElementById('histogram-plot-section').style.visibility = 'hidden';
                    cb_obj.tags = [window.outerWidth, window.outerHeight];
                """,
        )
        self._scenario_type_multi_choice.js_on_change("value", self._loading_js)
        self.planner_checkbox_group.js_on_change("active", self._loading_js)

        # Metric name multi choices
        self._metric_name_multi_choice = MultiChoice(
            css_classes=["metric-name-multi-choice"],
            name="histogram_metric_name_multi_choice",
            option_limit=default_multi_choice_style['option_limit'],
        )
        self._metric_name_multi_choice.on_change("value", self._metric_name_multi_choice_on_change)
        self._metric_name_multi_choice.js_on_change("value", self._loading_js)
        self._end_loading_js = CustomJS(
            args={},
            code="""
            document.getElementById('histogram-loading').style.visibility = 'hidden';
            document.getElementById('histogram-plot-section').style.visibility = 'visible';
        """,
        )
        self._bin_spinner = Spinner(
            mode='int',
            placeholder='Number of bins (default: 20)',
            low=default_spinner_style['low'],
            width=default_spinner_style['width'],
            css_classes=['histogram-bin-spinner'],
            name="histogram_bin_spinner",
        )
        self._bin_spinner.on_change("value", self._histogram_bin_spinner_on_change)
        self._bin_spinner.js_on_change("value", self._loading_js)

        self._default_div = Div(
            text=""" <p> No histogram results, please add more experiments or
        adjust the search filter.</p>""",
            css_classes=['histogram-default-div'],
            margin=default_div_style['margin'],
        )
        # Histogram plot frame.
        self._histogram_plots = column(
            self._default_div,
            css_classes=["histogram-plots"],
            name="histogram_plots",
        )
        self._histogram_plots.js_on_change("children", self._end_loading_js)
        self._histogram_figures: Optional[column] = None
        self._aggregated_data: Optional[histogram_data_type] = None
        self._histogram_edges: Optional[histogram_edges_data_type] = None
        self._plot_data: Dict[str, List[glyph]] = defaultdict(list)
        self._init_selection()

    @property
    def bin_spinner(self) -> Spinner:
        """Return a bin spinner."""
        return self._bin_spinner

    @property
    def scenario_type_multi_choice(self) -> MultiChoice:
        """Return scenario_type_multi_choice."""
        return self._scenario_type_multi_choice

    @property
    def metric_name_multi_choice(self) -> MultiChoice:
        """Return metric_name_multi_choice."""
        return self._metric_name_multi_choice

    @property
    def histogram_plots(self) -> column:
        """Return histogram_plots."""
        return self._histogram_plots

    def _click_planner_checkbox_group(self, attr: Any) -> None:
        """
        Click event handler for planner_checkbox_group.
        :param attr: Clicked attributes.
        """
        if not self._aggregated_data and not self._histogram_edges:
            return

        # Render histograms.
        self._histogram_figures = self._render_histograms()

        # Make sure the histogram upgrades at the last
        self._doc.add_next_tick_callback(self._update_histogram_layouts)

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

        self._init_selection()
        self._update_histograms()

    def _update_histogram_layouts(self) -> None:
        """Update histogram layouts."""
        self._histogram_plots.children[0] = layout(self._histogram_figures)

    def _update_histograms(self) -> None:
        """Update histograms."""
        # Aggregate data
        self._aggregated_data = self._aggregate_statistics()

        # Compute histogram edges
        self._histogram_edges = self._compute_histogram_edges()

        # Render histograms.
        self._histogram_figures = self._render_histograms()

        # Make sure the histogram upgrades at the last
        self._doc.add_next_tick_callback(self._update_histogram_layouts)

    def _histogram_bin_spinner_on_change(self, attr: str, old: int, new: int) -> None:
        """
        Helper function to change event in histogram spinner.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        self._bins = new

        # Compute histogram edges
        self._histogram_edges = self._compute_histogram_edges()

        # Render histograms.
        self._histogram_figures = self._render_histograms()

        # Make sure the histogram upgrades at the last
        self._doc.add_next_tick_callback(self._update_histogram_layouts)

    def _metric_name_multi_choice_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in histogram metric name.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        # Set up window width and height
        if self._metric_name_multi_choice.tags:
            self.window_width = self._metric_name_multi_choice.tags[0]
            self.window_height = self._metric_name_multi_choice.tags[1]
        self._update_histograms()

    def _scenario_type_multi_choice_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in histogram scenario type.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """
        # Set up window width and height
        if self._scenario_type_multi_choice.tags:
            self.window_width = self._scenario_type_multi_choice.tags[0]
            self.window_height = self.scenario_type_multi_choice.tags[1]
        self._update_histograms()

    def _init_selection(self) -> None:
        """Init histogram and scalar selection options."""
        # For planner checkbox
        planner_name_list: List[str] = []
        # Clean up
        self.planner_checkbox_group.labels = []
        self.planner_checkbox_group.active = []
        for index, metric_statistics_dataframes in enumerate(self.experiment_file_data.metric_statistics_dataframes):
            if index not in self._experiment_file_active_index:
                continue
            for metric_statistics_dataframe in metric_statistics_dataframes:
                planner_names = metric_statistics_dataframe.planner_names
                planner_name_list += planner_names

        sorted_planner_name_list = sorted(list(set(planner_name_list)))
        self.planner_checkbox_group.labels = sorted_planner_name_list
        self.planner_checkbox_group.active = [index for index in range(len(sorted_planner_name_list))]

        self._init_multi_search_criteria_selection(
            scenario_type_multi_choice=self._scenario_type_multi_choice,
            metric_name_multi_choice=self._metric_name_multi_choice,
        )

    def plot_vbar(
        self,
        histogram_figure_data: HistogramFigureData,
        counts: npt.NDArray[np.int64],
        category: List[str],
        planner_name: str,
        legend_label: str,
        color: str,
        scenario_names: List[str],
        x_values: List[str],
        width: float = 0.4,
    ) -> None:
        """
        Plot a vertical bar plot.
        :param histogram_figure_data: Figure class.
        :param counts: An array of counts for each category.
        :param category: A list of category (x-axis label).
        :param planner_name: Planner name.
        :param legend_label: Legend label.
        :param color: Legend color.
        :param scenario_names: A list of scenario names.
        :param x_values: X-axis values.
        :param width: Bar width.
        """
        y_values = deepcopy(counts)
        bottom: npt.NDArray[np.int64] = (
            np.zeros_like(counts)
            if histogram_figure_data.frequency_array is None
            else histogram_figure_data.frequency_array
        )
        count_position = counts > 0
        bottom_arrays: npt.NDArray[np.int64] = bottom * count_position
        top = counts + bottom_arrays

        data_source = ColumnDataSource(
            dict(
                x=category,
                top=top,
                bottom=bottom_arrays,
                y_values=y_values,
                x_values=x_values,
                scenario_names=scenario_names,
            )
        )
        figure_plot = histogram_figure_data.figure_plot
        vbar = figure_plot.vbar(
            x="x",
            top="top",
            bottom="bottom",
            fill_color=color,
            legend_label=legend_label,
            line_color=histogram_tab_style["quad_line_color"],
            fill_alpha=histogram_tab_style["quad_alpha"],
            line_alpha=histogram_tab_style["quad_alpha"],
            line_width=histogram_tab_style["quad_line_width"],
            width=width,
            source=data_source,
        )
        self._plot_data[planner_name].append(vbar)

        figure_plot.y_range.start = 0
        figure_plot.legend.background_fill_alpha = histogram_tab_style["plot_legend_background_fill_alpha"]
        figure_plot.legend.label_text_font_size = histogram_tab_style["plot_legend_label_text_font_size"]
        figure_plot.yaxis.axis_label = histogram_tab_style["plot_yaxis_axis_label"]
        figure_plot.grid.grid_line_color = histogram_tab_style["plot_grid_line_color"]

    def plot_histogram(
        self,
        histogram_figure_data: HistogramFigureData,
        hist: npt.NDArray[np.float64],
        edges: npt.NDArray[np.float64],
        planner_name: str,
        legend_label: str,
        color: str,
        scenario_names: List[str],
        x_values: List[str],
    ) -> None:
        """
        Plot a histogram.
        Reference from https://docs.bokeh.org/en/latest/docs/gallery/histogram.html.
        :param histogram_figure_data: Histogram figure data.
        :param hist: Histogram data.
        :param edges: Histogram bin data.
        :param planner_name: Planner name.
        :param legend_label: Legend label.
        :param color: Legend color.
        :param scenario_names: A list of scenario names.
        :param x_values: A list of x value names.
        """
        bottom: npt.NDArray[np.int64] = (
            np.zeros_like(hist)
            if histogram_figure_data.frequency_array is None
            else histogram_figure_data.frequency_array
        )
        hist_position = hist > 0
        bottom_arrays: npt.NDArray[np.int64] = bottom * hist_position
        top = hist + bottom_arrays
        data_source = ColumnDataSource(
            dict(
                top=top,
                bottom=bottom_arrays,
                left=edges[:-1],
                right=edges[1:],
                y_values=hist,
                x_values=x_values,
                scenario_names=scenario_names,
            )
        )
        figure_plot = histogram_figure_data.figure_plot
        quad = figure_plot.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            fill_color=color,
            line_color=histogram_tab_style["quad_line_color"],
            fill_alpha=histogram_tab_style["quad_alpha"],
            line_alpha=histogram_tab_style["quad_alpha"],
            legend_label=legend_label,
            line_width=histogram_tab_style["quad_line_width"],
            source=data_source,
        )

        self._plot_data[planner_name].append(quad)
        figure_plot.y_range.start = 0
        figure_plot.legend.background_fill_alpha = histogram_tab_style["plot_legend_background_fill_alpha"]
        figure_plot.legend.label_text_font_size = histogram_tab_style["plot_legend_label_text_font_size"]
        figure_plot.yaxis.axis_label = histogram_tab_style["plot_yaxis_axis_label"]
        figure_plot.grid.grid_line_color = histogram_tab_style["plot_grid_line_color"]

    def _render_histogram_plot(
        self,
        title: str,
        x_axis_label: str,
        x_range: Optional[List[str]] = None,
    ) -> HistogramFigureData:
        """
        Render a histogram plot.
        :param title: Title.
        :param x_axis_label: x axis label.
        :param x_range: A list of category data if specify.
        :return a figure.
        """
        tooltips = [("Frequency", "@y_values"), ("Values", "@x_values{safe}"), ("Scenarios", "@scenario_names{safe}")]
        statistic_figure = figure(
            background_fill_color=PLOT_PALETTE["background_white"],
            title=f"{title}",
            x_axis_label=f"{x_axis_label}",
            margin=histogram_tab_style["statistic_figure_margin"],
            width=self.plot_sizes[0],
            height=self.plot_sizes[1],
            x_range=x_range,
            output_backend="webgl",
            tooltips=tooltips,
        )

        statistic_figure.title.text_font_size = histogram_tab_style["statistic_figure_title_text_font_size"]
        statistic_figure.xaxis.axis_label_text_font_size = histogram_tab_style[
            "statistic_figure_xaxis_axis_label_text_font_size"
        ]
        statistic_figure.xaxis.major_label_text_font_size = histogram_tab_style[
            "statistic_figure_xaxis_major_label_text_font_size"
        ]
        statistic_figure.yaxis.axis_label_text_font_size = histogram_tab_style[
            "statistic_figure_yaxis_axis_label_text_font_size"
        ]
        statistic_figure.yaxis.major_label_text_font_size = histogram_tab_style[
            "statistic_figure_yaxis_major_label_text_font_size"
        ]
        statistic_figure.toolbar.logo = None

        return HistogramFigureData(figure_plot=statistic_figure)

    def _render_histogram_layout(self, histograms: histogram_figures_type) -> List[column]:
        """
        Render histogram layout.
        :param histograms: A dictionary of histogram names and their histograms.
        :return: A list of lists of figures (a list per row).
        """
        layouts = []
        for metric_statistics_name, statistics_data in histograms.items():
            title_div = Div(
                text=f"{metric_statistics_name}",
                style={"font-size": "10pt", "width": "100%", "font-weight": "bold"},
            )
            figures = [histogram_figure.figure_plot for statistic_name, histogram_figure in statistics_data.items()]
            grid_plot = gridplot(
                figures,
                ncols=self.get_plot_cols(plot_width=self.plot_sizes[0]),
                height=self.plot_sizes[1],
                toolbar_location="left",
            )
            grid_layout = column(title_div, grid_plot)
            layouts.append(grid_layout)

        return layouts

    def _aggregate_statistics(self) -> histogram_data_type:
        """
        Aggregate statistics data.
        :return A dictionary of metric names and their aggregated data.
        """
        data: histogram_data_type = defaultdict(list)
        scenario_types = self._scenario_type_multi_choice.value
        metric_choices = self._metric_name_multi_choice.value
        if not len(scenario_types):
            scenario_types = None
        else:
            scenario_types = tuple(scenario_types)

        if not scenario_types and not metric_choices:
            return data

        for index, metric_statistics_dataframes in enumerate(self.experiment_file_data.metric_statistics_dataframes):
            if index not in self._experiment_file_active_index:
                continue

            for metric_statistics_dataframe in metric_statistics_dataframes:
                if len(metric_choices) and metric_statistics_dataframe.metric_statistic_name not in metric_choices:
                    continue

                planner_names = metric_statistics_dataframe.planner_names
                for planner_name in planner_names:

                    data_frame = metric_statistics_dataframe.query_scenarios(
                        scenario_types=scenario_types, planner_names=tuple([planner_name])
                    )

                    if not len(data_frame):
                        continue

                    histogram_statistics_dict = {}
                    for statistic_name in metric_statistics_dataframe.statistic_names:
                        columns = [
                            f"{statistic_name}_stat_value",
                            f"{statistic_name}_stat_unit",
                            "scenario_name",
                            "log_name",
                        ]
                        statistic_data_frame = data_frame[columns]

                        # Value column index is 0, filter out inf values
                        values: npt.NDArray[np.float64] = np.asarray(statistic_data_frame.iloc[:, 0])
                        finite_states = np.isfinite(values)
                        values = values[finite_states]

                        scenarios: List[str] = list(np.asarray(statistic_data_frame.iloc[:, 2])[finite_states])
                        log_names: List[str] = list(np.asarray(statistic_data_frame.iloc[:, 3])[finite_states])
                        scenario_log_names: List[str] = []
                        for scenario, log_name in zip(scenarios, log_names):
                            scenario_log_names.append(scenario + " (" + log_name + ")")

                        if values.dtype != "bool":
                            values = np.round(values, 4)

                        # Unit column index is 1
                        unit = statistic_data_frame.iloc[0, 1]
                        histogram_statistics = HistogramStatistics(
                            unit=unit, values=values, scenarios=scenario_log_names
                        )
                        histogram_statistics_dict[statistic_name] = histogram_statistics

                    histogram_data = HistogramData(
                        experiment_index=index, planner_name=planner_name, statistics=histogram_statistics_dict
                    )
                    data[metric_statistics_dataframe.metric_statistic_name].append(histogram_data)

        return data

    def _compute_histogram_edges(self) -> histogram_edges_data_type:
        """
        Compute histogram edges across different planners in the same metric statistics.
        :return Histogram edge data.
        """
        histogram_edge_data: histogram_edges_data_type = {}
        if self._aggregated_data is None:
            return histogram_edge_data

        for metric_statistics_name, aggregated_histogram_data in self._aggregated_data.items():
            if metric_statistics_name not in histogram_edge_data:
                histogram_edge_data[metric_statistics_name] = {}
            edge_data: Dict[str, HistogramEdgeData] = {}
            for histogram_data in aggregated_histogram_data:
                for statistic_name, statistic in histogram_data.statistics.items():
                    unit = statistic.unit
                    if unit in ["bool", "boolean"]:
                        continue

                    if statistic_name not in edge_data:
                        edge_data[statistic_name] = HistogramEdgeData(unit=unit, values=statistic.values)
                    else:
                        edge_data[statistic_name].values = np.concatenate(
                            [edge_data[statistic_name].values, statistic.values]
                        )

            for statistic_name, statistic_edge_data in edge_data.items():
                unit = statistic_edge_data.unit
                if unit in ["count"]:
                    unique_values: npt.NDArray[np.float64] = np.unique(statistic_edge_data.values)
                    histogram_edge_data[metric_statistics_name][statistic_name] = (
                        None if not len(unique_values) else unique_values
                    )
                else:
                    _, edges = np.histogram(statistic_edge_data.values, bins=self._bins)
                    histogram_edge_data[metric_statistics_name][statistic_name] = None if not len(edges) else edges

        return histogram_edge_data

    def _plot_bool_histogram(
        self,
        histogram_figure_data: HistogramFigureData,
        values: npt.NDArray[np.float64],
        scenarios: List[str],
        planner_name: str,
        legend_name: str,
        color: str,
    ) -> None:
        """
        Plot boolean type of histograms.
        :param histogram_figure_data: Histogram figure data.
        :param values: An array of values.
        :param scenarios: A list of scenario names.
        :param planner_name: Planner name.
        :param legend_name: Legend name.
        :param color: Plot color.
        """
        # False and True
        num_true = sum(values)
        num_false = len(values) - num_true
        scenario_names: List[List[str]] = [[] for _ in range(2)]  # False and True bins only

        # Get scenario names
        for index, scenario in enumerate(scenarios):
            scenario_name_index = 1 if values[index] else 0
            if not self._max_scenario_names or len(scenario_names[scenario_name_index]) < self._max_scenario_names:
                scenario_names[scenario_name_index].append(scenario)
        scenario_names_flatten = ["<br>".join(names) if names else "" for names in scenario_names]
        counts: npt.NDArray[np.int64] = np.asarray([num_false, num_true])
        x_range = ["False", "True"]
        x_values = ["False", "True"]
        self.plot_vbar(
            histogram_figure_data=histogram_figure_data,
            category=x_range,
            counts=counts,
            planner_name=planner_name,
            legend_label=legend_name,
            color=color,
            scenario_names=scenario_names_flatten,
            x_values=x_values,
        )
        counts = np.asarray(counts)
        if histogram_figure_data.frequency_array is None:
            histogram_figure_data.frequency_array = deepcopy(counts)
        else:
            histogram_figure_data.frequency_array += counts

    def _plot_count_histogram(
        self,
        histogram_figure_data: HistogramFigureData,
        values: npt.NDArray[np.float64],
        scenarios: List[str],
        planner_name: str,
        legend_name: str,
        color: str,
        edges: npt.NDArray[np.float64],
    ) -> None:
        """
        Plot count type of histograms.
        :param histogram_figure_data: Histogram figure data.
        :param values: An array of values.
        :param scenarios: A list of scenario names.
        :param planner_name: Planner name.
        :param legend_name: Legend name.
        :param color: Plot color.
        :param edges: Count edges.
        """
        uniques: Any = np.unique(values, return_inverse=True)
        unique_values: npt.NDArray[np.float64] = uniques[0]
        unique_index: npt.NDArray[np.int64] = uniques[1]

        counts = {value: 0 for value in edges}
        bin_count = np.bincount(unique_index)

        for index, count_value in enumerate(bin_count):
            counts[unique_values[index]] = count_value

        # Get scenario names
        scenario_names: List[List[str]] = [[] for _ in range(len(counts))]
        for index, bin_index in enumerate(unique_index):
            if not self._max_scenario_names or len(scenario_names[bin_index]) < self._max_scenario_names:
                scenario_names[bin_index].append(scenarios[index])
        scenario_names_flatten = ["<br>".join(names) if names else "" for names in scenario_names]
        category = [str(key) for key in counts.keys()]
        count_values: npt.NDArray[np.int64] = np.asarray(list(counts.values()))

        self.plot_vbar(
            histogram_figure_data=histogram_figure_data,
            category=category,
            counts=count_values,
            planner_name=planner_name,
            legend_label=legend_name,
            color=color,
            scenario_names=scenario_names_flatten,
            width=0.1,
            x_values=category,
        )
        if histogram_figure_data.frequency_array is None:
            histogram_figure_data.frequency_array = deepcopy(count_values)
        else:
            histogram_figure_data.frequency_array += count_values

    def _plot_bin_histogram(
        self,
        histogram_figure_data: HistogramFigureData,
        values: npt.NDArray[np.float64],
        scenarios: List[str],
        planner_name: str,
        legend_name: str,
        color: str,
        edges: npt.NDArray[np.float64],
    ) -> None:
        """
        Plot bin type of histograms.
        :param histogram_figure_data: Histogram figure data.
        :param values: An array of values.
        :param scenarios: A list of scenario names.
        :param planner_name: Planner name.
        :param legend_name: Legend name.
        :param color: Plot color.
        :param edges: Histogram bin edges.
        """
        hist, bins = np.histogram(values, bins=edges)
        value_bin_index: npt.NDArray[np.int64] = np.asarray(np.digitize(values, bins=bins[:-1]))

        # Get scenario names
        scenario_names: List[List[str]] = [[] for _ in range(len(hist))]
        for index, bin_index in enumerate(value_bin_index):
            if not self._max_scenario_names or len(scenario_names[bin_index - 1]) < self._max_scenario_names:
                scenario_names[bin_index - 1].append(scenarios[index])
        scenario_names_flatten = ["<br>".join(names) if names else "" for names in scenario_names]

        # Get x_values
        bins = np.round(bins, 4)
        x_values = [str(value) + ' - ' + str(bins[index + 1]) for index, value in enumerate(bins[:-1])]
        self.plot_histogram(
            histogram_figure_data=histogram_figure_data,
            planner_name=planner_name,
            legend_label=legend_name,
            hist=hist,
            edges=edges,
            color=color,
            scenario_names=scenario_names_flatten,
            x_values=x_values,
        )
        if histogram_figure_data.frequency_array is None:
            histogram_figure_data.frequency_array = deepcopy(hist)
        else:
            histogram_figure_data.frequency_array += hist

    def _draw_histogram_data(self) -> histogram_figures_type:
        """
        Draw histogram data based on aggregated data.
        :return A dictionary of metric names and theirs histograms.
        """
        histograms: histogram_figures_type = defaultdict()
        if self._aggregated_data is None or self._histogram_edges is None:
            return histograms

        for metric_statistics_name, aggregated_histogram_data in self._aggregated_data.items():
            if metric_statistics_name not in histograms:
                histograms[metric_statistics_name] = {}

            for histogram_data in aggregated_histogram_data:
                legend_name = (
                    histogram_data.planner_name + f" ({self.get_file_path_last_name(histogram_data.experiment_index)})"
                )
                if histogram_data.planner_name not in self.enable_planner_names:
                    continue

                color = self.experiment_file_data.file_path_colors[histogram_data.experiment_index][
                    histogram_data.planner_name
                ]
                for statistic_name, statistic in histogram_data.statistics.items():
                    unit = statistic.unit

                    if statistic_name not in histograms[metric_statistics_name]:
                        x_range: Optional[List[str]] = None
                        # Boolean type of data
                        if unit in ["bool", "boolean"]:
                            x_range = ["False", "True"]
                        elif unit in ["count"]:
                            data = self._histogram_edges[metric_statistics_name].get(statistic_name, None)
                            assert data is not None, f"Count edge data for {statistic_name} cannot be None!"
                            x_range = [str(count) for count in data]
                        histograms[metric_statistics_name][statistic_name] = self._render_histogram_plot(
                            title=statistic_name, x_axis_label=unit, x_range=x_range
                        )
                    histogram_figure_data = histograms[metric_statistics_name][statistic_name]
                    values = statistic.values

                    # Boolean type of data
                    if unit in ["bool", "boolean"]:
                        self._plot_bool_histogram(
                            histogram_figure_data=histogram_figure_data,
                            values=values,
                            scenarios=statistic.scenarios,
                            planner_name=histogram_data.planner_name,
                            legend_name=legend_name,
                            color=color,
                        )
                    else:
                        edges = self._histogram_edges[metric_statistics_name][statistic_name]
                        if edges is None:
                            continue

                        # Count type
                        if unit in ["count"]:
                            self._plot_count_histogram(
                                histogram_figure_data=histogram_figure_data,
                                values=values,
                                scenarios=statistic.scenarios,
                                planner_name=histogram_data.planner_name,
                                legend_name=legend_name,
                                color=color,
                                edges=edges,
                            )
                        else:
                            self._plot_bin_histogram(
                                histogram_figure_data=histogram_figure_data,
                                values=values,
                                scenarios=statistic.scenarios,
                                planner_name=histogram_data.planner_name,
                                legend_name=legend_name,
                                color=color,
                                edges=edges,
                            )

        # Sort
        sorted_histograms = {key: histograms[key] for key in sorted(histograms.keys(), reverse=False)}
        return sorted_histograms

    def _render_histograms(self) -> List[column]:
        """
        Render histograms across all scenarios based on a scenario type.
        :return: A list of lists of figures (a list per row).
        """
        histograms = self._draw_histogram_data()
        layouts = self._render_histogram_layout(histograms)
        # Empty plot
        if not layouts:
            layouts = [
                column(
                    self._default_div,
                    css_classes=["histogram-plots"],
                    width=800,
                    name="histogram_plots",
                )
            ]
        return layouts
