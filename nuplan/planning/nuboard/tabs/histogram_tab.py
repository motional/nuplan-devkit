import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from bokeh.document.document import Document
from bokeh.layouts import column, gridplot, layout
from bokeh.models import Button, ColumnDataSource, Div, FactorRange, HoverTool, MultiChoice, Spinner, glyph
from bokeh.plotting import figure

from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.tabs.config.histogram_tab_config import (
    HISTOGRAM_TAB_DEFAULT_NUMBER_COLS,
    PLANNER_CHECKBOX_GROUP_NAME,
    SCENARIO_TYPE_SCORE_HISTOGRAM_NAME,
    HistogramData,
    HistogramDataType,
    HistogramEdgeData,
    HistogramEdgesDataType,
    HistogramFigureData,
    HistogramFigureDataType,
    HistogramStatistics,
    HistogramTabBinSpinnerConfig,
    HistogramTabDefaultDivConfig,
    HistogramTabFigureGridPlotStyleConfig,
    HistogramTabFigureStyleConfig,
    HistogramTabFigureTitleDivStyleConfig,
    HistogramTabHistogramBarStyleConfig,
    HistogramTabMetricNameMultiChoiceConfig,
    HistogramTabModalQueryButtonConfig,
    HistogramTabPlotConfig,
    HistogramTabScenarioTypeMultiChoiceConfig,
)
from nuplan.planning.nuboard.tabs.js_code.histogram_tab_js_code import (
    HistogramTabLoadingEndJSCode,
    HistogramTabLoadingJSCode,
    HistogramTabUpdateWindowsSizeJSCode,
)

logger = logging.getLogger(__name__)


class HistogramTab(BaseTab):
    """Histogram tab in nuBoard."""

    def __init__(
        self,
        doc: Document,
        experiment_file_data: ExperimentFileData,
        bins: int = HistogramTabBinSpinnerConfig.default_bins,
        max_scenario_names: int = 20,
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
        self.planner_checkbox_group.name = PLANNER_CHECKBOX_GROUP_NAME
        self.planner_checkbox_group.js_on_change("active", HistogramTabLoadingJSCode.get_js_code())

        # Scenario type multi choices
        self._scenario_type_multi_choice = MultiChoice(**HistogramTabScenarioTypeMultiChoiceConfig.get_config())
        self._scenario_type_multi_choice.on_change("value", self._scenario_type_multi_choice_on_change)
        self._scenario_type_multi_choice.js_on_change("value", HistogramTabUpdateWindowsSizeJSCode.get_js_code())

        # Metric name multi choices
        self._metric_name_multi_choice = MultiChoice(**HistogramTabMetricNameMultiChoiceConfig.get_config())
        self._metric_name_multi_choice.on_change("value", self._metric_name_multi_choice_on_change)
        self._metric_name_multi_choice.js_on_change("value", HistogramTabUpdateWindowsSizeJSCode.get_js_code())

        self._bin_spinner = Spinner(**HistogramTabBinSpinnerConfig.get_config())
        self._histogram_modal_query_btn = Button(**HistogramTabModalQueryButtonConfig.get_config())
        self._histogram_modal_query_btn.js_on_click(HistogramTabLoadingJSCode.get_js_code())
        self._histogram_modal_query_btn.on_click(self._setting_modal_query_button_on_click)

        self._default_div = Div(**HistogramTabDefaultDivConfig.get_config())
        # Histogram plot frame.
        self._histogram_plots = column(self._default_div, **HistogramTabPlotConfig.get_config())
        self._histogram_plots.js_on_change("children", HistogramTabLoadingEndJSCode.get_js_code())
        self._histogram_figures: Optional[column] = None
        self._aggregated_data: Optional[HistogramDataType] = None
        self._histogram_edges: Optional[HistogramEdgesDataType] = None
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

    @property
    def histogram_modal_query_btn(self) -> Button:
        """Return histogram modal query button."""
        return self._histogram_modal_query_btn

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

        scenario_type_score_data = self._aggregate_scenario_type_score_histogram()

        self._aggregated_data.update(scenario_type_score_data)

        # Compute histogram edges
        self._histogram_edges = self._compute_histogram_edges()

        # Render histograms.
        self._histogram_figures = self._render_histograms()

        # Make sure the histogram upgrades at the last
        self._doc.add_next_tick_callback(self._update_histogram_layouts)

    def _setting_modal_query_button_on_click(self) -> None:
        """Setting modal query button on click helper function."""
        if self._metric_name_multi_choice.tags:
            self.window_width = self._metric_name_multi_choice.tags[0]
            self.window_height = self._metric_name_multi_choice.tags[1]

        if self._bin_spinner.value:
            self._bins = self._bin_spinner.value
        self._update_histograms()

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

    def _adjust_plot_width_size(self, n_bins: int) -> int:
        """
        Adjust plot width size based on number of bins.
        :param n_bins: Number of bins.
        :return Width size of a histogram plot.
        """
        base_plot_width: int = self.plot_sizes[0]
        if n_bins < 20:
            return base_plot_width
        # Increase the width of 50 for every number of bins 20
        width_multiplier_factor: int = (n_bins // 20) * 100
        width_size: int = min(
            base_plot_width + width_multiplier_factor, HistogramTabFigureStyleConfig.maximum_plot_width
        )
        return width_size

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
        histogram_file_name: Optional[str] = None,
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
        :param histogram_file_name: Histogram file name for the histogram data.
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
        histogram_file_names = [histogram_file_name] * len(top)
        data_source = ColumnDataSource(
            dict(
                x=category,
                top=top,
                bottom=bottom_arrays,
                y_values=y_values,
                x_values=x_values,
                scenario_names=scenario_names,
                histogram_file_name=histogram_file_names,
            )
        )
        figure_plot = histogram_figure_data.figure_plot
        vbar = figure_plot.vbar(
            x="x",
            top="top",
            bottom="bottom",
            fill_color=color,
            legend_label=legend_label,
            width=width,
            source=data_source,
            **HistogramTabHistogramBarStyleConfig.get_config(),
        )
        self._plot_data[planner_name].append(vbar)
        HistogramTabHistogramBarStyleConfig.update_histogram_bar_figure_style(histogram_figure=figure_plot)

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
        histogram_file_name: Optional[str] = None,
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
        :param histogram_file_name: Histogram file name for the histogram data.
        """
        bottom: npt.NDArray[np.int64] = (
            np.zeros_like(hist)
            if histogram_figure_data.frequency_array is None
            else histogram_figure_data.frequency_array
        )
        hist_position = hist > 0
        bottom_arrays: npt.NDArray[np.int64] = bottom * hist_position
        top = hist + bottom_arrays
        histogram_file_names = [histogram_file_name] * len(top)
        data_source = ColumnDataSource(
            dict(
                top=top,
                bottom=bottom_arrays,
                left=edges[:-1],
                right=edges[1:],
                y_values=hist,
                x_values=x_values,
                scenario_names=scenario_names,
                histogram_file_name=histogram_file_names,
            )
        )
        figure_plot = histogram_figure_data.figure_plot
        quad = figure_plot.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            fill_color=color,
            legend_label=legend_label,
            **HistogramTabHistogramBarStyleConfig.get_config(),
            source=data_source,
        )

        self._plot_data[planner_name].append(quad)
        HistogramTabHistogramBarStyleConfig.update_histogram_bar_figure_style(histogram_figure=figure_plot)

    def _render_histogram_plot(
        self,
        title: str,
        x_axis_label: str,
        x_range: Optional[Union[List[str], FactorRange]] = None,
        histogram_file_name: Optional[str] = None,
    ) -> HistogramFigureData:
        """
        Render a histogram plot.
        :param title: Title.
        :param x_axis_label: x-axis label.
        :param x_range: A list of category data if specified.
        :param histogram_file_name: Histogram file name for the histogram plot.
        :return a figure.
        """
        if x_range is None:
            len_plot_width = 1
        elif isinstance(x_range, list):
            len_plot_width = len(x_range)
        else:
            len_plot_width = len(x_range.factors)

        plot_width = self._adjust_plot_width_size(n_bins=len_plot_width)
        tooltips = [("Frequency", "@y_values"), ("Values", "@x_values{safe}"), ("Scenarios", "@scenario_names{safe}")]
        if histogram_file_name:
            tooltips.append(("File", "@histogram_file_name"))

        hover_tool = HoverTool(tooltips=tooltips, point_policy="follow_mouse")
        statistic_figure = figure(
            **HistogramTabFigureStyleConfig.get_config(
                title=title, x_axis_label=x_axis_label, width=plot_width, height=self.plot_sizes[1], x_range=x_range
            ),
            tools=["pan", "wheel_zoom", "save", "reset", hover_tool],
        )
        HistogramTabFigureStyleConfig.update_histogram_figure_style(histogram_figure=statistic_figure)
        return HistogramFigureData(figure_plot=statistic_figure)

    def _render_histogram_layout(self, histograms: HistogramFigureDataType) -> List[column]:
        """
        Render histogram layout.
        :param histograms: A dictionary of histogram names and their histograms.
        :return: A list of lists of figures (a list per row).
        """
        layouts = []
        ncols = self.get_plot_cols(plot_width=self.plot_sizes[0], default_ncols=HISTOGRAM_TAB_DEFAULT_NUMBER_COLS)
        for metric_statistics_name, statistics_data in histograms.items():
            title_div = Div(**HistogramTabFigureTitleDivStyleConfig.get_config(title=metric_statistics_name))
            figures = [histogram_figure.figure_plot for statistic_name, histogram_figure in statistics_data.items()]
            grid_plot = gridplot(
                figures,
                **HistogramTabFigureGridPlotStyleConfig.get_config(ncols=ncols, height=self.plot_sizes[1]),
            )
            grid_layout = column(title_div, grid_plot)
            layouts.append(grid_layout)

        return layouts

    @staticmethod
    def _get_scenario_score_type_score_histogram_data(
        experiment_index: int,
        data: HistogramDataType,
        scenario_type_score_histogram_statistics_dict: Dict[str, Dict[str, List[Tuple[float, str]]]],
        metric_aggregator_file_name: str,
    ) -> HistogramDataType:
        """
        Get histogram data from scenario type score histogram statistics.
        :param experiment_index: Experiment index.
        :param data: Histogram data.
        :param scenario_type_score_histogram_statistics_dict: Statistics dictionary with scenario types and their
        scores.
        :param metric_aggregator_file_name: Metric aggregator file name.
        :return Scenario type histogram data.
        """
        for planner_name, scenario_type_info in scenario_type_score_histogram_statistics_dict.items():
            histogram_statistics_dict = {}
            for scenario_type_name, scenario_type_scores in scenario_type_info.items():
                values = np.round(
                    np.asarray([score[0] for score in scenario_type_scores], dtype=np.float64),
                    HistogramTabFigureStyleConfig.decimal_places,
                )
                scenario_log_names = [score[1] for score in scenario_type_scores]
                histogram_statistics = HistogramStatistics(unit='scores', values=values, scenarios=scenario_log_names)
                histogram_statistics_dict[scenario_type_name] = histogram_statistics

            histogram_data = HistogramData(
                experiment_index=experiment_index,
                planner_name=planner_name,
                statistics=histogram_statistics_dict,
                histogram_file_name=metric_aggregator_file_name,
            )
            data[SCENARIO_TYPE_SCORE_HISTOGRAM_NAME].append(histogram_data)
        return data

    def _aggregate_scenario_type_score_histogram(self) -> HistogramDataType:
        """
        Aggregate metric aggregator data.
        :return: A dictionary of metric aggregator names and their metric scores.
        """
        data: HistogramDataType = defaultdict(list)
        selected_scenario_types = self._scenario_type_multi_choice.value

        # Loop through all metric aggregators
        for index, metric_aggregator_dataframes in enumerate(self.experiment_file_data.metric_aggregator_dataframes):
            if index not in self._experiment_file_active_index:
                continue
            for metric_aggregator_file_name, metric_aggregator_dataframe in metric_aggregator_dataframes.items():
                # Planner name: scenario type: List[(score, scenario)]
                scenario_type_score_histogram_statistics_dict: Dict[
                    str, Dict[str, List[Tuple[float, str]]]
                ] = defaultdict(lambda: defaultdict(list))
                # scenario names

                # Iterate through rows
                for _, row_data in metric_aggregator_dataframe.iterrows():
                    num_scenarios = row_data["num_scenarios"]
                    if not np.isnan(num_scenarios):
                        continue
                    scenario_name = row_data['scenario'] + " (" + row_data['log_name'] + ")"
                    if 'all' in selected_scenario_types:
                        scenario_type_score_histogram_statistics_dict[row_data["planner_name"]]['all'].append(
                            (row_data["score"], scenario_name)
                        )
                    if 'all' in selected_scenario_types or row_data['scenario_type'] in selected_scenario_types:
                        scenario_type_score_histogram_statistics_dict[row_data["planner_name"]][
                            row_data["scenario_type"]
                        ].append((row_data["score"], scenario_name))
                self._get_scenario_score_type_score_histogram_data(
                    experiment_index=index,
                    data=data,
                    scenario_type_score_histogram_statistics_dict=scenario_type_score_histogram_statistics_dict,
                    metric_aggregator_file_name=metric_aggregator_file_name,
                )

        return data

    def _aggregate_statistics(self) -> HistogramDataType:
        """
        Aggregate statistics data.
        :return A dictionary of metric names and their aggregated data.
        """
        data: HistogramDataType = defaultdict(list)
        scenario_types = self._scenario_type_multi_choice.value
        metric_choices = self._metric_name_multi_choice.value
        if not len(scenario_types) and not len(metric_choices):
            return data

        if 'all' in scenario_types:
            scenario_types = None
        else:
            scenario_types = tuple(scenario_types)

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

                        # Value column index is 0, make all inf values to be nan
                        values: npt.NDArray[np.float64] = np.asarray(statistic_data_frame.iloc[:, 0], dtype=np.float64)
                        infinite_states = np.isinf(values)
                        values[infinite_states] = np.nan

                        scenarios: List[str] = list(np.asarray(statistic_data_frame.iloc[:, 2]))
                        log_names: List[str] = list(np.asarray(statistic_data_frame.iloc[:, 3]))
                        scenario_log_names: List[str] = []
                        for scenario, log_name in zip(scenarios, log_names):
                            scenario_log_names.append(scenario + " (" + log_name + ")")

                        if values.dtype != "bool":
                            values = np.round(values, HistogramTabFigureStyleConfig.decimal_places)

                        # Unit column index is 1
                        unit = statistic_data_frame.iloc[0, 1]
                        if unit is None:
                            unit = 'value'
                        histogram_statistics = HistogramStatistics(
                            unit=unit, values=values, scenarios=scenario_log_names
                        )
                        histogram_statistics_dict[statistic_name] = histogram_statistics

                    histogram_data = HistogramData(
                        experiment_index=index, planner_name=planner_name, statistics=histogram_statistics_dict
                    )
                    data[metric_statistics_dataframe.metric_statistic_name].append(histogram_data)

        return data

    def _compute_histogram_edges(self) -> HistogramEdgesDataType:
        """
        Compute histogram edges across different planners in the same metric statistics.
        :return Histogram edge data.
        """
        histogram_edge_data: HistogramEdgesDataType = {}
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
                    # Remove nan
                    finite_values = statistic_edge_data.values[~np.isnan(statistic_edge_data.values)]
                    _, edges = np.histogram(finite_values, bins=self._bins)
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
        histogram_file_name: Optional[str] = None,
    ) -> None:
        """
        Plot boolean type of histograms.
        :param histogram_figure_data: Histogram figure data.
        :param values: An array of values.
        :param scenarios: A list of scenario names.
        :param planner_name: Planner name.
        :param legend_name: Legend name.
        :param color: Plot color.
        :param histogram_file_name: Histogram file name for the histogram data.
        """
        # False and True
        num_true = np.nansum(values)
        num_false = len(values[values == 0])
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
            histogram_file_name=histogram_file_name,
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
        histogram_file_name: Optional[str] = None,
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
        :param histogram_file_name: Histogram file name for the histogram data.
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
            histogram_file_name=histogram_file_name,
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
        histogram_file_name: Optional[str] = None,
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
        :param histogram_file_name: Histogram file name for the histogram data.
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
        bins = np.round(bins, HistogramTabFigureStyleConfig.decimal_places)
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
            histogram_file_name=histogram_file_name,
        )
        if histogram_figure_data.frequency_array is None:
            histogram_figure_data.frequency_array = deepcopy(hist)
        else:
            histogram_figure_data.frequency_array += hist

    @staticmethod
    def _get_histogram_plot_x_range(unit: str, data: npt.NDArray[np.float64]) -> Union[List[str], FactorRange]:
        """
        Get Histogram x_range based on unit and data.
        :param unit: Histogram unit.
        :param data: Histogram data.
        :return x_range in histogram plot.
        """
        x_range = None
        # Boolean type of data
        if unit in ["bool", "boolean"]:
            x_range = ["False", "True"]
        elif unit in ["count"]:
            x_range = [str(count) for count in data]
        return x_range

    def _draw_histogram_data(self) -> HistogramFigureDataType:
        """
        Draw histogram data based on aggregated data.
        :return A dictionary of metric names and theirs histograms.
        """
        histograms: HistogramFigureDataType = defaultdict()
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
                    data: npt.NDArray[np.float64] = np.unique(
                        self._histogram_edges[metric_statistics_name].get(statistic_name, None)
                    )
                    assert data is not None, f"Count edge data for {statistic_name} cannot be None!"
                    if statistic_name not in histograms[metric_statistics_name]:
                        # Boolean type of data
                        x_range = self._get_histogram_plot_x_range(unit=unit, data=data)
                        histograms[metric_statistics_name][statistic_name] = self._render_histogram_plot(
                            title=statistic_name,
                            x_axis_label=unit,
                            x_range=x_range,
                            histogram_file_name=histogram_data.histogram_file_name,
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
                            histogram_file_name=histogram_data.histogram_file_name,
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
                                histogram_file_name=histogram_data.histogram_file_name,
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
                                histogram_file_name=histogram_data.histogram_file_name,
                            )

        # Make scenario type score always the first one
        sorted_histograms = {}
        if SCENARIO_TYPE_SCORE_HISTOGRAM_NAME in histograms:
            sorted_histograms[SCENARIO_TYPE_SCORE_HISTOGRAM_NAME] = histograms[SCENARIO_TYPE_SCORE_HISTOGRAM_NAME]
        # Sort
        sorted_histogram_keys = sorted(
            (key for key in histograms.keys() if key != SCENARIO_TYPE_SCORE_HISTOGRAM_NAME), reverse=False
        )
        sorted_histograms.update({key: histograms[key] for key in sorted_histogram_keys})
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
                    self._default_div, width=HistogramTabPlotConfig.default_width, **HistogramTabPlotConfig.get_config()
                )
            ]
        return layouts
