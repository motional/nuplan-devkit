import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from bokeh.document.document import Document
from bokeh.layouts import column, gridplot, layout, row
from bokeh.models import Div, MultiChoice, Panel
from bokeh.plotting import figure
from nuplan.planning.metrics.metric_file import MetricFile
from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.style import MOTIONAL_PALETTE, PLOT_PALETTE, histogram_tab_style

logger = logging.getLogger(__name__)


histogram_data_type = Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]
histogram_figure_type = Dict[str, Dict[str, Dict[str, figure]]]


class HistogramTab(BaseTab):

    def __init__(self,
                 doc: Document,
                 file_paths: List[NuBoardFile]):
        """
        Histogram tab for metric results.
        :param file_paths: Path to a metric pickle file.
        """

        super().__init__(doc=doc, file_paths=file_paths)

        # UI.
        self._scenario_type_multi_choice = MultiChoice(title="Scenarios", margin=self.search_criteria_margin)
        self._scenario_type_multi_choice.on_change("value", self._scenario_type_multi_choice_on_change)
        self._metric_name_multi_choice = MultiChoice(title="Metrics", margin=self.search_criteria_margin)
        self._metric_name_multi_choice.on_change("value", self._metric_name_multi_choice_on_change)
        search_criteria = column(self.search_criteria_title,
                                 self._scenario_type_multi_choice, self._metric_name_multi_choice,
                                 css_classes=["search-criteria-panel"], height=self.search_criteria_height)

        self._histogram_plot_title = Div(text="""<h3 style='margin-left: 18px;
        margin-top: 20px;'>Histograms</h3>""")

        # Histogram plot frame.
        histogram_plots = column(self._histogram_plot_title, Div(),
                                 css_classes=["histogram-panel"],
                                 sizing_mode="scale_height",
                                 height=self.plot_frame_sizes[1])
        self._plot_layout = row([search_criteria, histogram_plots], css_classes=["histogram-tab"])
        self._panel = Panel(title="Histograms", child=self._plot_layout)
        self._init_selection()

    def file_paths_on_change(self, file_paths: List[NuBoardFile]) -> None:
        """ Update if file_path is changed. """
        self._file_paths = file_paths

        self._scenario_type_multi_choice.value = []
        self._scenario_type_multi_choice.options = []
        self._metric_name_multi_choice.value = []
        self._metric_name_multi_choice.options = []

        self.load_metric_files(reset=True)
        self.load_simulation_files(reset=True)
        self._init_selection()

    def _update_histograms(self) -> None:
        """ Update histograms. """

        # Render histograms.
        histogram_figures = self._render_histograms()

        plot_col = self._plot_layout.children[1]
        plot_col.children[1] = layout(histogram_figures)

    def _metric_name_multi_choice_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in histogram metric name.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """

        self._update_histograms()

    def _scenario_type_multi_choice_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in histogram scenario type.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """

        self._update_histograms()

    def _init_selection(self) -> None:
        """ Init histogram and scalar selection options. """

        self._init_multi_search_criteria_selection(scenario_type_multi_choice=self._scenario_type_multi_choice,
                                                   metric_name_multi_choice=self._metric_name_multi_choice)

    @staticmethod
    def plot_vbar(p: figure, counts: List[int], category: List[str], legend_label: str, color: str) -> figure:
        """
        Plot a vertical bar plot.
        :param p: Figure class.
        :param counts: A list of counts for each category.
        :param category: A list of category (x-axis label).
        :param legend_label: Legend label.
        :param color: Legend color.
        :return a figure.
        """

        p.vbar(x=category,
               top=counts,
               fill_color=color,
               legend_label=legend_label,
               line_color=histogram_tab_style['quad_line_color'],
               fill_alpha=histogram_tab_style['quad_alpha'],
               line_alpha=histogram_tab_style['quad_alpha'],
               line_width=histogram_tab_style['quad_line_width'],
               width=0.4)

        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        p.legend.background_fill_alpha = histogram_tab_style['plot_legend_background_fill_alpha']
        p.legend.label_text_font_size = histogram_tab_style['plot_legend_label_text_font_size']
        p.yaxis.axis_label = histogram_tab_style['plot_yaxis_axis_label']
        p.grid.grid_line_color = histogram_tab_style['plot_grid_line_color']
        return p

    @staticmethod
    def plot_histogram(p: figure, hist: npt.NDArray[np.float32], edges: npt.NDArray[np.float32],
                       x: List[float], legend_label: str, color: str,
                       pdf: Optional[npt.NDArray[np.float32]] = None,
                       cdf: Optional[npt.NDArray[np.float32]] = None) -> figure:
        """
        Plot a histogram.
        Reference from https://docs.bokeh.org/en/latest/docs/gallery/histogram.html.
        :param p: Figure class.
        :param hist: Histogram data.
        :param edges: Histogram bin data.
        :param x: x-axis data.
        :param legend_label: Legend label.
        :param color: Legend color.
        :param pdf: Set to draw a pdf curve.
        :param cdf: Set to draw a cdf curve.
        :return a figure.
        """

        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color=color, line_color=histogram_tab_style['quad_line_color'],
               fill_alpha=histogram_tab_style['quad_alpha'],
               line_alpha=histogram_tab_style['quad_alpha'],
               legend_label=legend_label,
               line_width=histogram_tab_style['quad_line_width'])
        if pdf is not None:
            p.line(x,
                   pdf,
                   line_color=PLOT_PALETTE['chart_green'],
                   line_width=histogram_tab_style['pdf_line_width'],
                   alpha=histogram_tab_style['pdf_alpha'],
                   legend_label="PDF")
        if cdf is not None:
            p.line(x,
                   cdf,
                   line_color=PLOT_PALETTE['chart_yellow'],
                   line_width=histogram_tab_style['cdf_line_width'],
                   alpha=histogram_tab_style['cdf_alpha'],
                   legend_label="CDF")

        p.y_range.start = 0
        p.legend.background_fill_alpha = histogram_tab_style['plot_legend_background_fill_alpha']
        p.legend.label_text_font_size = histogram_tab_style['plot_legend_label_text_font_size']
        p.yaxis.axis_label = histogram_tab_style['plot_yaxis_axis_label']
        p.grid.grid_line_color = histogram_tab_style['plot_grid_line_color']
        return p

    def _render_histogram_plot(self,
                               title: str,
                               x_axis_label: str,
                               x_range: Optional[List[str]] = None,
                               ) -> figure:
        """
        Render a histogram plot.
        :param title: Title.
        :param x_axis_label: x axis label.
        :param x_range: A list of category data if specify.
        :return a figure.
        """

        statistic_figure = figure(background_fill_color=PLOT_PALETTE['background_white'], title=f"{title}",
                                  x_axis_label=f"{x_axis_label}", margin=histogram_tab_style['statistic_figure_margin'],
                                  height=self.plot_sizes[1], x_range=x_range
                                  )

        statistic_figure.title.text_font_size = histogram_tab_style['statistic_figure_title_text_font_size']
        statistic_figure.xaxis.axis_label_text_font_size = \
            histogram_tab_style['statistic_figure_xaxis_axis_label_text_font_size']
        statistic_figure.xaxis.major_label_text_font_size = \
            histogram_tab_style['statistic_figure_xaxis_major_label_text_font_size']
        statistic_figure.yaxis.axis_label_text_font_size = \
            histogram_tab_style['statistic_figure_yaxis_axis_label_text_font_size']
        statistic_figure.yaxis.major_label_text_font_size = \
            histogram_tab_style['statistic_figure_yaxis_major_label_text_font_size']
        statistic_figure.toolbar.logo = None

        return statistic_figure

    def _render_histogram_layout(self, histograms: Dict[str, figure]) -> List[column]:
        """
        Render histogram layout.
        :param histograms: A dictionary of histogram names and their histograms.
        :return: A list of lists of figures (a list per row).
        """

        layouts = []
        for metric_name, metric_result_figures in histograms.items():
            title_div = Div(text=f"{metric_name}", style={'font-size': '16pt', 'width': '20%'},
                            margin=histogram_tab_style['histogram_title_div_margin'])
            figures = []
            for metric_result_name, statistic_figures in metric_result_figures.items():
                for statistic_name, statistic_figure in statistic_figures.items():
                    figures.append(statistic_figure)
            grid_plot = gridplot(figures, ncols=self.plot_cols)
            grid_layout = column(title_div, grid_plot)
            layouts.append(grid_layout)

        return layouts

    @staticmethod
    def _aggregate_statistics(metric_files: List[MetricFile]) -> histogram_data_type:
        """
        Aggregate statistics data.
        :param metric_files: A list of loaded metric files.
        :return A dictionary of metric names and their aggregated data.
        """

        data: histogram_data_type = {}
        for metric_file in metric_files:
            key = metric_file.key
            metric_name = key.metric_name
            planner_name = key.planner_name

            if metric_name not in data:
                data[metric_name] = {}

            for statistic_group_name, metric_statistics in metric_file.metric_statistics.items():
                if statistic_group_name not in data[metric_name]:
                    data[metric_name][statistic_group_name] = {}

                for metric_statistic in metric_statistics:
                    for statistic_type, statistic in metric_statistic.statistics.items():
                        if statistic.name not in data[metric_name][statistic_group_name]:
                            data[metric_name][statistic_group_name][statistic.name] = {
                                'unit': statistic.unit,
                                'planners': defaultdict(list)
                            }

                        data_statistics = data[metric_name][statistic_group_name][statistic.name]['planners']
                        data_statistics[planner_name].append(statistic.value)

        return data

    def _draw_histogram_data(self, data: histogram_data_type) -> histogram_figure_type:
        """
        Draw histogram data based on aggregated data.
        :param data: Aggregated statistical data.
        :return A dictionary of metric names and theirs histograms.
        """

        histograms: histogram_figure_type = {}
        color_keys = list(MOTIONAL_PALETTE.keys())
        for metric_name, metric_results in data.items():

            if metric_name not in histograms:
                histograms[metric_name] = defaultdict(dict)

            for metric_result_name, statistics in metric_results.items():
                if metric_result_name not in histograms[metric_name]:
                    histograms[metric_name][metric_result_name] = {}

                for statistic_name, statistic in statistics.items():
                    unit = statistic['unit']

                    # Boolean type of data
                    x_range: Optional[List[str]] = None
                    if unit in ['bool', 'boolean']:
                        x_range = ['False', 'True']

                    if statistic_name not in histograms[metric_name][metric_result_name]:
                        statistic_figure = self._render_histogram_plot(title=statistic_name, x_axis_label=unit,
                                                                       x_range=x_range)
                        histograms[metric_name][metric_result_name][statistic_name] = statistic_figure
                    else:
                        statistic_figure = histograms[metric_name][metric_result_name][statistic_name]

                    planners = statistic['planners']

                    # Boolean type of data
                    if unit in ['bool', 'boolean']:
                        # False and True
                        for color_index, (planner_name, values) in enumerate(planners.items()):
                            num_true = sum(values)
                            num_false = len(values) - num_true
                            counts = [num_false, num_true]
                            color = MOTIONAL_PALETTE[color_keys[color_index]]
                            x_range = ['False', 'True']
                            self.plot_vbar(p=statistic_figure,
                                           category=x_range,
                                           counts=counts,
                                           legend_label=f"{planner_name}", color=color)
                    else:
                        # Compute edges.
                        edge_values = []
                        for planner_name, value in planners.items():
                            edge_values += value

                        # Filter out inf values.
                        edge_values = np.asarray(edge_values)
                        edge_values = edge_values[np.isfinite(edge_values)]
                        _, edges = np.histogram(edge_values, bins=20)

                        if len(edge_values) == 0:
                            continue

                        for color_index, (planner_name, values) in enumerate(planners.items()):
                            hist, _ = np.histogram(values, bins=edges)
                            color = MOTIONAL_PALETTE[color_keys[color_index]]
                            self.plot_histogram(p=statistic_figure,
                                                legend_label=f"{planner_name}", hist=hist,
                                                edges=edges, x=values, color=color)

        return histograms

    def _render_histograms(self) -> List[column]:
        """
        Render histograms across all scenarios based on a scenario type.
        :return: A list of lists of figures (a list per row).
        """

        selected_keys = []
        for key in self.metric_scenario_keys:
            if key.scenario_type in self._scenario_type_multi_choice.value:
                # If no metric result selected.
                if len(self._metric_name_multi_choice.value) == 0 or \
                        key.metric_result_name in self._metric_name_multi_choice.value:
                    selected_keys.append(key)

        if len(selected_keys) == 0:
            return []

        metric_files = [self._read_metric_file(key.file) for key in selected_keys]
        data = self._aggregate_statistics(metric_files=metric_files)
        histograms = self._draw_histogram_data(data=data)

        # Sort
        sorted_histograms = {key: histograms[key] for key in sorted(histograms.keys(), reverse=False)}
        layouts = self._render_histogram_layout(sorted_histograms)

        return layouts
