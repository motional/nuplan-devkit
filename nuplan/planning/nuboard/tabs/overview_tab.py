import logging
from typing import Any, Dict, List

import numpy as np
from bokeh.document.document import Document
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, DataTable, Div, MultiChoice, Panel, TableColumn
from nuplan.planning.metrics.metric_file import MetricFile
from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.style import overview_tab_style

logger = logging.getLogger(__name__)


time_series_data_type = Dict[str, Dict[str, Dict[str, List[float]]]]


class OverviewTab(BaseTab):

    def __init__(self,
                 doc: Document,
                 metric_categories: List[str],
                 file_paths: List[NuBoardFile]):
        """
        Metric board to render metrics.
        :param file_paths: Path to a metric pickle file.
        """

        super().__init__(doc=doc, file_paths=file_paths)
        self._metric_categories = metric_categories

        # UI.
        self._scenario_type_multi_choice = MultiChoice(title="Scenarios", margin=self.search_criteria_margin)
        self._scenario_type_multi_choice.on_change("value", self._scenario_type_multi_choice_on_change)
        self._metric_name_multi_choice = MultiChoice(title="Metrics", margin=self.search_criteria_margin)
        self._metric_name_multi_choice.on_change("value", self._metric_name_multi_choice_on_change)

        search_criteria = column(self.search_criteria_title,
                                 self._scenario_type_multi_choice, self._metric_name_multi_choice,
                                 css_classes=["search-criteria-panel"], height=self.search_criteria_height)

        self._overview_title = Div(text="""<h3 style='margin-left: 18px;
        margin-top: 20px;'>Overview</h3>""")

        self._tables = {}
        table_columns = []
        for metric_category in self._metric_categories:
            title = Div(text=f"""<h4 style='margin-left: 18px;
            margin-top: 20px; width: 200px;'>{metric_category}</h4>""")
            source = ColumnDataSource(dict())

            columns = [TableColumn(field='metrics', title='Metric')]
            self._tables[metric_category] = DataTable(source=source, columns=columns,
                                                      margin=overview_tab_style['table_margins'],
                                                      width=overview_tab_style['table_width'],
                                                      height=overview_tab_style['table_height'],
                                                      css_classes=["overview-table"])
            table_columns += [title, self._tables[metric_category]]
        overview_table = column([self._overview_title] + table_columns,
                                css_classes=["overview-panel"],
                                height=self.plot_frame_sizes[1], sizing_mode="scale_height")
        self._plot_layout = row([search_criteria, overview_table], css_classes=["overview-tab"])
        self._panel = Panel(title="Overview", child=self._plot_layout)

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

    def _metric_name_multi_choice_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in histogram metric name.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """

        self._compute_time_series_data_overview()

    def _scenario_type_multi_choice_on_change(self, attr: str, old: str, new: str) -> None:
        """
        Helper function to change event in histogram scenario type.
        :param attr: Attribute.
        :param old: Old value.
        :param new: New value.
        """

        self._compute_time_series_data_overview()

    def _init_selection(self) -> None:
        """ Init histogram and scalar selection options. """

        self._init_multi_search_criteria_selection(scenario_type_multi_choice=self._scenario_type_multi_choice,
                                                   metric_name_multi_choice=self._metric_name_multi_choice)

    @staticmethod
    def _aggregate_time_series(metric_files: List[MetricFile]) -> time_series_data_type:
        """
        Aggregate time series data.
        :param metric_files: A list of loaded metric files.
        :return A dictionary of metric names and their aggregated time series data.
        """

        data: time_series_data_type = {}
        for metric_file in metric_files:
            key = metric_file.key
            metric_name = key.metric_name
            planner_name = key.planner_name

            for statistic_group_name, metric_statistics in metric_file.metric_statistics.items():
                for metric_statistic in metric_statistics:
                    time_series = metric_statistic.time_series
                    if time_series is None:
                        continue

                    metric_category = metric_statistic.metric_category

                    if metric_category not in data:
                        data[metric_category] = {}
                    if planner_name not in data[metric_category]:
                        data[metric_category][planner_name] = {}

                    if metric_name not in data[metric_category][planner_name]:
                        data[metric_category][planner_name][metric_name] = []

                    data[metric_category][planner_name][metric_name] += time_series.values

        return data

    def _update_overview_tables(self, time_series_data: time_series_data_type) -> None:
        """
        Update overview tables based on the aggregated time series data.
        :param time_series_data: Aggregated time series data.
        """

        for metric_category in self._metric_categories:
            data_sources: Dict[str, Any] = {}
            table_columns = [
                TableColumn(field="metric_names", title="Metrics")
            ]
            planners = time_series_data.get(metric_category, None)
            if planners is not None:
                for planner_name, metrics in planners.items():
                    statistics_data: Dict[str, float] = {}
                    for metric_name, metric_value in metrics.items():
                        # filter out inf values and catch empty array
                        metric_value = np.asarray(metric_value)
                        metric_value = metric_value[np.isfinite(metric_value)]
                        if len(metric_value):
                            mean_value = np.round(np.nanmean(metric_value), 4)
                        else:
                            mean_value = 0.0
                        statistics_data[f"mean_{metric_name}"] = mean_value

                        if len(metric_value):
                            max_value = np.round(np.nanmax(metric_value), 4)
                        else:
                            max_value = 0.0
                        statistics_data[f"max_{metric_name}"] = max_value

                    data_sources[f'{planner_name}_values'] = list(statistics_data.values())
                    table_column = TableColumn(field=f'{planner_name}_values', title=planner_name)
                    table_columns.append(table_column)
                    data_sources["metric_names"] = list(statistics_data.keys())
            self._tables[metric_category].columns = table_columns
            self._tables[metric_category].source.data = data_sources

    def _compute_time_series_data_overview(self) -> None:
        """
        Render overview of time series data across all scenarios based on a scenario type or metric names.
        :return: A list of lists of figures (a list per row).
        """

        selected_keys = []
        for key in self.metric_scenario_keys:
            if key.scenario_type in self._scenario_type_multi_choice.value:

                # If no metric result selected.
                if len(self._metric_name_multi_choice.value) == 0 or \
                        key.metric_result_name in self._metric_name_multi_choice.value:
                    selected_keys.append(key)

        metric_results = [self._read_metric_file(key.file) for key in selected_keys]
        time_series_data = self._aggregate_time_series(metric_files=metric_results)
        self._update_overview_tables(time_series_data=time_series_data)
