import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from bokeh.document.document import Document
from bokeh.models import ColumnDataSource, DataTable, TableColumn

from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.tabs.config.overview_tab_config import (
    OVERVIEW_PLANNER_CHECKBOX_GROUP_NAME,
    OverviewAggregatorData,
    OverviewTabDataTableConfig,
    OverviewTabDefaultDataSourceDictConfig,
    OverviewTabExperimentTableColumnConfig,
    OverviewTabPlannerTableColumnConfig,
    OverviewTabScenarioTypeTableColumnConfig,
)

logger = logging.getLogger(__name__)


class OverviewTab(BaseTab):
    """Overview tab in nuBoard."""

    def __init__(self, doc: Document, experiment_file_data: ExperimentFileData):
        """
        Overview tab to visualize general metric results about simulation.
        :param doc: Bokeh HTML document.
        :param experiment_file_data: Experiment file data.
        """
        super().__init__(doc=doc, experiment_file_data=experiment_file_data)

        self._aggregator_metric_data: Dict[str, List[OverviewAggregatorData]] = {}

        self._default_datasource_dict = dict(**OverviewTabDefaultDataSourceDictConfig.get_config())
        self._default_datasource = ColumnDataSource(data=self._default_datasource_dict)

        self._default_columns = [
            TableColumn(**OverviewTabExperimentTableColumnConfig.get_config()),
            TableColumn(**OverviewTabScenarioTypeTableColumnConfig.get_config()),
            TableColumn(**OverviewTabPlannerTableColumnConfig.get_config()),
        ]

        self.table = DataTable(
            source=self._default_datasource, columns=self._default_columns, **OverviewTabDataTableConfig.get_config()
        )

        self.planner_checkbox_group.name = OVERVIEW_PLANNER_CHECKBOX_GROUP_NAME

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

        self._overview_on_change()

    def _click_planner_checkbox_group(self, attr: Any) -> None:
        """
        Click event handler for planner_checkbox_group.
        :param attr: Clicked attributes.
        """
        self._update_overview_table(self._aggregator_metric_data)

    def _overview_on_change(self) -> None:
        """Callback when metric search bar changes."""
        # Aggregator metric aggregator data
        self._aggregator_metric_data = self._aggregate_metric_aggregator()
        self._update_overview_table(data=self._aggregator_metric_data)

    def _aggregate_metric_aggregator(self) -> Dict[str, List[OverviewAggregatorData]]:
        """
        Aggregate metric aggregator data.
        :return: A dictionary of metric aggregator names and their metric scores.
        """
        data: Dict[str, List[OverviewAggregatorData]] = defaultdict(list)
        # For planner checkbox
        planner_names: List[str] = []
        # Loop through all metric aggregators
        for index, metric_aggregator_dataframes in enumerate(self.experiment_file_data.metric_aggregator_dataframes):
            if index not in self._experiment_file_active_index:
                continue
            for metric_aggregator_filename, metric_aggregator_dataframe in metric_aggregator_dataframes.items():
                # Iterate through rows
                for _, row_data in metric_aggregator_dataframe.iterrows():
                    num_scenarios = row_data["num_scenarios"]
                    if not num_scenarios or np.isnan(num_scenarios):
                        continue

                    aggregator_type = row_data["aggregator_type"]
                    planner_name = row_data["planner_name"]
                    scenario_type = row_data["scenario_type"]
                    metric_score = row_data["score"]
                    # Add aggregator data to the data dictionary, the key is
                    data[metric_aggregator_filename].append(
                        OverviewAggregatorData(
                            aggregator_type=aggregator_type,
                            planner_name=planner_name,
                            scenario_type=scenario_type,
                            num_scenarios=int(num_scenarios),
                            score=metric_score,
                            aggregator_file_name=metric_aggregator_filename,
                        )
                    )
                    planner_names.append(planner_name)

        sorted_planner_names = sorted(list(set(planner_names)))
        self.planner_checkbox_group.labels = sorted_planner_names
        self.planner_checkbox_group.active = [index for index in range(len(sorted_planner_names))]
        return data

    def _update_overview_table(self, data: Dict[str, List[OverviewAggregatorData]]) -> None:
        """
        Update overview table with the new metric aggregator data.
        :param data: Metric aggregator data.
        """
        # Get all planner names
        planner_names = sorted(
            list(
                {
                    metric_aggregator_data.planner_name
                    for _, metric_aggregator_data_list in data.items()
                    for metric_aggregator_data in metric_aggregator_data_list
                    if metric_aggregator_data.planner_name in self.enable_planner_names
                }
            )
        )
        planner_name_columns: Dict[str, List[Any]] = {planner_name: [] for planner_name in planner_names}
        metric_aggregator_files: List[str] = []
        scenario_types: List[str] = []
        for metric_file, metric_aggregator_data_list in data.items():
            metric_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
            initial_planner_names: List[str] = []
            for metric_aggregator_data in metric_aggregator_data_list:
                initial_planner_names.append(metric_aggregator_data.planner_name)
                if metric_aggregator_data.planner_name not in self.enable_planner_names:
                    continue

                metric_scores[metric_aggregator_data.planner_name][metric_aggregator_data.scenario_type] = [
                    np.round(metric_aggregator_data.score, 4),
                    metric_aggregator_data.num_scenarios,
                ]

            initial_planner_names = list(set(initial_planner_names))
            if metric_scores:
                metric_aggregator_files += [metric_file] + [''] * (
                    (len(metric_aggregator_data_list) // len(initial_planner_names)) - 1
                )

            # Sorted by scenario type
            metric_aggregator_file_scenario_types: List[str] = []
            for planner_name, values in metric_scores.items():
                sorted_metric_scores: Dict[str, List[float]] = dict(
                    sorted(list(values.items()), key=lambda item: item[0])
                )
                # Make sure final_score is always the first
                sorted_final_metric_scores = {
                    f'all ({sorted_metric_scores["final_score"][1]})': sorted_metric_scores['final_score'][0]
                }
                sorted_final_metric_scores.update(
                    {
                        f'{scenario_type} ({score[1]})': score[0]
                        for scenario_type, score in sorted_metric_scores.items()
                        if scenario_type != 'final_score'
                    }
                )
                metric_file_scenario_types = list(sorted_final_metric_scores.keys())
                metric_file_scenario_type_scores = list(sorted_final_metric_scores.values())
                planner_name_columns[planner_name] += metric_file_scenario_type_scores
                if not metric_aggregator_file_scenario_types:
                    metric_aggregator_file_scenario_types += metric_file_scenario_types
            scenario_types += metric_aggregator_file_scenario_types

            # Fill in empty planners
            for planner_name in planner_names:
                if planner_name not in metric_scores:
                    empty_scenario_scores = ['-'] * len(metric_aggregator_file_scenario_types)
                    planner_name_columns[planner_name] += empty_scenario_scores
        if planner_name_columns:
            data_sources: Dict[str, List[Any]] = {
                'experiment': metric_aggregator_files,
                'scenario_type': scenario_types,
            }
            data_sources.update(planner_name_columns)

            # Make planner columns
            planner_table_columns = [
                TableColumn(field=planner_name, title=f'Evaluation Score {index+1}: {planner_name}', sortable=False)
                for index, planner_name in enumerate(planner_name_columns.keys())
            ]
            columns = [
                TableColumn(**OverviewTabExperimentTableColumnConfig.get_config()),
                TableColumn(**OverviewTabScenarioTypeTableColumnConfig.get_config()),
            ]
            columns += planner_table_columns
            self.table.columns = columns
            self.table.source.data = data_sources
        else:
            self.table.columns = self._default_columns
            self.table.source.data = self._default_datasource_dict
