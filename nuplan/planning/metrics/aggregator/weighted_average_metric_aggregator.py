from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.metrics.aggregator.abstract_metric_aggregator import AbstractMetricAggregator
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame

logger = logging.getLogger(__name__)
metric_aggregator_dict_column = Dict[str, Dict[str, Any]]


class WeightedAverageMetricAggregator(AbstractMetricAggregator):
    """Metric aggregator by implementing weighted sum."""

    def __init__(
        self,
        name: str,
        metric_weights: Dict[str, float],
        file_name: str,
        aggregator_save_path: Path,
        multiple_metrics: List[str],
        challenge_name: Optional[str] = None,
    ):
        """
        Initializes the WeightedAverageMetricAggregator class.
        :param name: Metric aggregator name.
        :param metric_weights: Weights for each metric. Default would be 1.0.
        :param file_name: Saved file name.
        :param aggregator_save_path: Save path for this aggregated parquet file.
        :param multiple_metrics: A list if metric names used in multiple factor when computing scenario scores.
        :param challenge_name: Optional, name of the challenge the metrics refer to, if set will be part of the
        output file name and path.
        """
        self._name = name
        self._metric_weights = metric_weights
        self._file_name = file_name

        if not self._file_name.endswith('.parquet'):
            self._file_name += '.parquet'

        self._aggregator_save_path = aggregator_save_path

        self._challenge_name = challenge_name

        if not is_s3_path(self._aggregator_save_path):
            self._aggregator_save_path.mkdir(exist_ok=True, parents=True)
        self._aggregator_type = 'weighted_average'
        self._multiple_metrics = multiple_metrics
        self._parquet_file = self._aggregator_save_path / self._file_name

        #  Aggregated metric dataframe: <scenario types, metric scores>
        self._aggregated_metric_dataframe: Optional[pandas.DataFrame] = None

    @property
    def aggregated_metric_dataframe(self) -> Optional[pandas.DataFrame]:
        """Return the aggregated metric dataframe."""
        return self._aggregated_metric_dataframe

    @property
    def name(self) -> str:
        """
        Return the metric aggregator name.
        :return the metric aggregator name.
        """
        return self._name

    @property
    def final_metric_score(self) -> Optional[float]:
        """Return the final metric score."""
        # The last column (bottom right) should always be the final metric score from this aggregator
        if self._aggregated_metric_dataframe is not None:
            return self._aggregated_metric_dataframe.iloc[-1, -1]  # type: ignore
        else:
            logger.warning('The metric not yet aggregated.')
            return None

    def _get_metric_weight(self, metric_name: str) -> float:
        """
        Get metric weights.
        :param metric_name: The metric name.
        :return Weight for the metric.
        """
        # Check any metric name specified in the metric weight dictionary, if no then the weight would be
        # default weight or 1.0
        weight: Optional[float] = self._metric_weights.get(metric_name, None)
        metric_weight = self._metric_weights.get('default', 1.0) if weight is None else weight

        return metric_weight

    def _compute_scenario_score(self, scenario_metric_columns: metric_aggregator_dict_column) -> None:
        """
        Compute scenario scores.
        :param scenario_metric_columns: Scenario metric column in the format of {scenario_names: {metric_column:
        value}}.
        """
        excluded_columns = ['log_name', 'planner_name', 'aggregator_type', 'scenario_type', 'num_scenarios', 'score']
        for scenario_name, columns in scenario_metric_columns.items():
            metric_scores = 0.0
            sum_weights = 0.0
            multiple_factor = 1.0
            for column_key, column_value in columns.items():
                # Skip if column key is excluded or the value is None
                if column_key in excluded_columns or column_value is None:
                    continue
                if self._multiple_metrics and column_key in self._multiple_metrics:
                    multiple_factor *= column_value
                else:
                    weight = self._get_metric_weight(metric_name=column_key)
                    assert column_value is not None, f"Metric: {column_key} value should not be None!"
                    assert weight is not None, f"Metric: {column_key} weight " f"should not be None!"
                    sum_weights += weight
                    metric_scores += weight * column_value
            weighted_average_score = metric_scores / sum_weights if sum_weights else 0.0
            final_score = multiple_factor * weighted_average_score
            scenario_metric_columns[scenario_name]['score'] = final_score

    @staticmethod
    def _group_scenario_type_metric(
        scenario_metric_columns: metric_aggregator_dict_column,
    ) -> metric_aggregator_dict_column:
        """
        Group scenario type metric columns in the format of {scenario_type: {metric_columns: value}}.
        :param scenario_metric_columns: Scenario metric columns in the format of {scenario_name: {metric_columns:
        value}}.
        :return Metric columns based on scenario type.
        """
        # Transform to scenario_type: {}
        scenario_type_dicts: metric_aggregator_dict_column = defaultdict(lambda: defaultdict(list))
        total_scenarios = len(scenario_metric_columns)
        for scenario_name, columns in scenario_metric_columns.items():
            scenario_type = columns['scenario_type']
            scenario_type_dicts[scenario_type]['scenario_name'].append(scenario_name)
            for column_key, column_value in columns.items():
                scenario_type_dicts[scenario_type][column_key].append(column_value)

        # Column get only first index value
        common_columns = ['planner_name', 'aggregator_type', 'scenario_type']
        # Excluded columns
        excluded_columns = ['scenario_name']
        scenario_type_metric_columns: metric_aggregator_dict_column = defaultdict(lambda: defaultdict())
        for scenario_type, columns in scenario_type_dicts.items():
            for key, values in columns.items():
                if key in excluded_columns:
                    continue
                elif key in common_columns:
                    scenario_type_metric_columns[scenario_type][key] = values[0]
                elif key == 'log_name':
                    scenario_type_metric_columns[scenario_type][key] = None
                # Handle the column num_scenarios specifically
                elif key == 'num_scenarios':
                    scenario_type_metric_columns[scenario_type]['num_scenarios'] = len(values)
                else:
                    available_values: npt.NDArray[np.float64] = np.asarray(
                        [value for value in values if value is not None]
                    )
                    value: Optional[float] = float(np.sum(available_values)) if available_values.size > 0 else None
                    # Handle the column final_score specifically for weighted average by the number of scenarios later
                    if key == 'score' and value is not None:
                        # scenario_type_score = sum(scenario_final_score) / total number of scenarios
                        score_value: float = value / len(values) if total_scenarios else 0.0
                        scenario_type_metric_columns[scenario_type][key] = score_value
                    else:
                        scenario_type_metric_columns[scenario_type][key] = value

        return scenario_type_metric_columns

    @staticmethod
    def _group_final_score_metric(
        scenario_type_metric_columns: metric_aggregator_dict_column,
    ) -> metric_aggregator_dict_column:
        """
        Compute a final score based on a group of scenario types.
        :param scenario_type_metric_columns: Scenario type metric columns in the format of {scenario_type:
        {metric_column: value}}.
        :return A dictionary of final score in the format of {'final_score': {metric_column: value}}.
        """
        # Transform to final_score: {}
        final_score_dicts: metric_aggregator_dict_column = defaultdict(lambda: defaultdict(list))
        for scenario_type, columns in scenario_type_metric_columns.items():
            for column_key, column_value in columns.items():
                final_score_dicts['final_score'][column_key].append(column_value)

        final_score_metric_columns: metric_aggregator_dict_column = defaultdict(lambda: defaultdict())
        total_scenarios = sum(final_score_dicts['final_score']['num_scenarios'])
        # Column get only first index value
        common_columns = ['planner_name', 'aggregator_type']
        for final_score_column_name, columns in final_score_dicts.items():
            for key, values in columns.items():
                if key == 'scenario_type':
                    final_score_metric_columns[final_score_column_name][key] = 'final_score'
                elif key == 'log_name':
                    final_score_metric_columns[final_score_column_name][key] = None
                elif key in common_columns:
                    final_score_metric_columns[final_score_column_name][key] = values[0]
                elif key == 'num_scenarios':
                    final_score_metric_columns[final_score_column_name][key] = total_scenarios
                else:
                    available_values: List[float] = []
                    if key == 'score':
                        for value, num_scenario in zip(values, columns['num_scenarios']):
                            if value is not None:
                                available_values.append(value * num_scenario)
                    else:
                        available_values = [value for value in values if value is not None]

                    if not available_values:
                        total_values = None
                    else:
                        available_value_array: npt.NDArray[np.float64] = np.asarray(available_values)
                        total_values = np.sum(available_value_array) / total_scenarios

                    final_score_metric_columns[final_score_column_name][key] = total_values
        return final_score_metric_columns

    def _group_scenario_metrics(
        self, metric_dataframes: Dict[str, MetricStatisticsDataFrame], planner_name: str
    ) -> metric_aggregator_dict_column:
        """
        Group scenario metrics in the format of {scenario_name: {metric_column: value}}.
        :param metric_dataframes: A dict of metric dataframes.
        :param planner_name: A planner name.
        :return Dictionary column format in metric aggregator in {scenario_name: {metric_column: value}}.
        """
        metric_names = sorted(list(metric_dataframes.keys()))
        columns = {
            column: None
            for column in ['log_name', 'planner_name', 'aggregator_type', 'scenario_type', 'num_scenarios']
            + metric_names
            + ['score']
        }

        scenario_metric_columns: metric_aggregator_dict_column = {}
        for metric_name, metric_dataframe in metric_dataframes.items():
            # Get only dataframe with the planner names only
            dataframe = metric_dataframe.query_scenarios(planner_names=tuple([planner_name]))
            for _, data in dataframe.iterrows():
                scenario_name = data.get('scenario_name')
                if scenario_name not in scenario_metric_columns:
                    scenario_metric_columns[scenario_name] = deepcopy(columns)
                scenario_type = data['scenario_type']
                scenario_metric_columns[scenario_name]['log_name'] = data['log_name']
                scenario_metric_columns[scenario_name]['planner_name'] = data['planner_name']
                scenario_metric_columns[scenario_name]['scenario_type'] = scenario_type
                scenario_metric_columns[scenario_name]['aggregator_type'] = self._aggregator_type
                scenario_metric_columns[scenario_name][metric_name] = data['metric_score']
        return scenario_metric_columns

    def __call__(self, metric_dataframes: Dict[str, MetricStatisticsDataFrame]) -> None:
        """
        Run an aggregator to generate an aggregated parquet file.
        :param metric_dataframes: A dictionary of metric name and dataframe.
        """
        # Get all planner names
        planner_names = sorted(
            list(
                {
                    planner_name
                    for metric_statistic_dataframe in metric_dataframes.values()
                    for planner_name in metric_statistic_dataframe.planner_names
                }
            )
        )

        weighted_average_dataframe_columns: Dict[str, List[Any]] = dict()
        for planner_name in planner_names:

            metric_names = sorted(list(metric_dataframes.keys())) + ['score']
            dataframe_columns: Dict[str, List[Any]] = {
                'scenario': [],
                'log_name': [],
                'scenario_type': [],
                'num_scenarios': [],
                'planner_name': [],
                'aggregator_type': [],
            }
            metric_name_columns: Dict[str, List[float]] = {metric_name: [] for metric_name in metric_names}
            dataframe_columns.update(metric_name_columns)
            # Group scenario metrics
            scenario_metric_columns = self._group_scenario_metrics(
                metric_dataframes=metric_dataframes, planner_name=planner_name
            )

            # Compute scenario scores
            self._compute_scenario_score(scenario_metric_columns=scenario_metric_columns)
            # Get metric columns based on scenario types
            scenario_type_metric_columns = self._group_scenario_type_metric(
                scenario_metric_columns=scenario_metric_columns
            )

            # Compute a final score based on scenario types
            scenario_type_final_metric_columns = self._group_final_score_metric(
                scenario_type_metric_columns=scenario_type_metric_columns
            )

            # Append scenario type metric columns to scenario metric columns
            scenario_metric_columns.update(scenario_type_metric_columns)

            # Append final_score metric columns to scenario metric columns
            scenario_metric_columns.update(scenario_type_final_metric_columns)

            # Arrange columns into dict format
            for scenario_name, columns in scenario_metric_columns.items():
                dataframe_columns['scenario'].append(scenario_name)
                for key, value in columns.items():
                    dataframe_columns[key].append(value)
            if not weighted_average_dataframe_columns:
                weighted_average_dataframe_columns.update(dataframe_columns)
            else:
                for column_name, value in weighted_average_dataframe_columns.items():
                    value += dataframe_columns[column_name]
        # Convert to pandas dataframe
        self._aggregated_metric_dataframe = pandas.DataFrame(data=weighted_average_dataframe_columns)

        # Save to a parquet file
        self._save_parquet(dataframe=self._aggregated_metric_dataframe, save_path=self._parquet_file)

    def read_parquet(self) -> None:
        """Read a parquet file."""
        self._aggregated_metric_dataframe = pandas.read_parquet(self._parquet_file)

    @property
    def parquet_file(self) -> Path:
        """Inherited, see superclass"""
        return self._parquet_file

    @property
    def challenge(self) -> Optional[str]:
        """Inherited, see superclass"""
        return self._challenge_name
