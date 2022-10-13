import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from bokeh.models import FactorRange

from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
from nuplan.planning.nuboard.tabs.config.histogram_tab_config import (
    HistogramConstantConfig,
    HistogramData,
    HistogramEdgeData,
    HistogramStatistics,
    HistogramTabFigureStyleConfig,
)

logger = logging.getLogger(__name__)


def _extract_metric_statistics_dict(
    metric_statistics_dataframe: MetricStatisticsDataFrame,
    planner_name: str,
    scenario_types: Optional[Tuple[str]] = None,
) -> Optional[Dict[str, HistogramStatistics]]:
    """
    Extract metric statistics dataframe and aggregate them into histogram data in a dictionary of
    {metric_statistics_name: HistogramStatistics}.
    :param metric_statistics_dataframe: Metric statistics dataframe.
    :param scenario_types: A tuple of scenario types to search from the metric statistics dataframe.
    :param planner_name: A planner name to search from the metric statistics dataframe.
    :return A dictionary of {metric_statistic_name: HistogramStatistics}.
    """
    histogram_statistics_dict = {}
    data_frame: pd.DataFrame = metric_statistics_dataframe.query_scenarios(
        scenario_types=scenario_types, planner_names=(planner_name,)
    )
    if not len(data_frame):
        return None
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
        unit = statistic_data_frame.iloc[0, 1] or 'value'
        histogram_statistics = HistogramStatistics(unit=unit, values=values, scenarios=scenario_log_names)
        histogram_statistics_dict[statistic_name] = histogram_statistics
    return histogram_statistics_dict


def _extract_scenario_score_type_histogram_statistics_dict(
    scenario_type_info: Dict[str, List[Tuple[float, str]]]
) -> Dict[str, HistogramStatistics]:
    """
    Extract histogram statistics dict from a dictionary of {scenario_type_name: [(scenario_score, scenario_name)]}
    :param scenario_type_info: A dict of {scenario_type_name: [(scenario_score, scenario_name)]}
    :returns A dictionary of scenario type names and histogram statistics.
    """
    histogram_statistics_dict = {}
    for scenario_type_name, scenario_type_scores in scenario_type_info.items():
        values = np.round(
            np.asarray([score[0] for score in scenario_type_scores], dtype=np.float64),
            HistogramTabFigureStyleConfig.decimal_places,
        )
        scenario_log_names = [score[1] for score in scenario_type_scores]
        histogram_statistics = HistogramStatistics(unit='scores', values=values, scenarios=scenario_log_names)
        histogram_statistics_dict[scenario_type_name] = histogram_statistics
    return histogram_statistics_dict


def _extract_metric_aggregator_scenario_type_score_data(
    metric_aggregator_dataframe: pd.DataFrame, scenario_types: List[str]
) -> HistogramConstantConfig.HistogramScenarioTypeScoreStatisticType:
    """
    Extract scenario type scores from metric aggregator dataframe and transform it to a histogram form in a
    dictionary of {'planner_name': {'scenario_type': [scenario_score, scenario_name]}}.
    :param metric_aggregator_dataframe: Metric aggregator dataframe.
    :param scenario_types: List of selected scenario types.
    :return {'planner_name': {'scenario_type': [scenario_score, scenario_name]}}.
    """
    # Planner name: scenario type: List[(score, scenario)]
    scenario_type_score_histogram_statistics_dict: HistogramConstantConfig.HistogramScenarioTypeScoreStatisticType = (
        defaultdict(lambda: defaultdict(list))
    )

    # Iterate through rows
    for _, row_data in metric_aggregator_dataframe.iterrows():
        num_scenarios = row_data["num_scenarios"]
        if not np.isnan(num_scenarios):
            continue
        scenario_name = row_data['scenario'] + " (" + row_data['log_name'] + ")"
        if 'all' in scenario_types:
            scenario_type_score_histogram_statistics_dict[row_data["planner_name"]]['all'].append(
                (row_data["score"], scenario_name)
            )
        if 'all' in scenario_types or row_data['scenario_type'] in scenario_types:
            scenario_type_score_histogram_statistics_dict[row_data["planner_name"]][row_data["scenario_type"]].append(
                (row_data["score"], scenario_name)
            )
    return scenario_type_score_histogram_statistics_dict


def extract_scenario_score_type_score_histogram_data(
    histogram_file_name: str,
    metric_aggregator_dataframe_index: int,
    scenario_type_score_histogram_statistics_dict: HistogramConstantConfig.HistogramScenarioTypeScoreStatisticType,
) -> List[HistogramData]:
    """
    Get histogram data from scenario type score histogram statistics.
    :param histogram_file_name: Histogram file name.
    :param metric_aggregator_dataframe_index: Metric aggregator dataframe index.
    :param scenario_type_score_histogram_statistics_dict: Statistics dictionary with scenario types and their
    scores.
    :return Scenario type histogram data.
    """
    histogram_data_list: List[HistogramData] = []
    for planner_name, scenario_type_info in scenario_type_score_histogram_statistics_dict.items():
        histogram_statistics_dict = _extract_scenario_score_type_histogram_statistics_dict(
            scenario_type_info=scenario_type_info
        )
        histogram_data = HistogramData(
            experiment_index=metric_aggregator_dataframe_index,
            planner_name=planner_name,
            statistics=histogram_statistics_dict,
            histogram_file_name=histogram_file_name,
        )
        histogram_data_list.append(histogram_data)
    return histogram_data_list


def aggregate_metric_aggregator_dataframe_histogram_data(
    dataframe_file_name: str,
    metric_aggregator_dataframe: pd.DataFrame,
    metric_aggregator_dataframe_index: int,
    scenario_types: List[str],
) -> List[HistogramData]:
    """
    Aggregate metric statistics dataframe data for histograms.
    :param dataframe_file_name: Dataframe file name.
    :param metric_aggregator_dataframe: Metric aggregator dataframe.
    :param metric_aggregator_dataframe_index: Metric aggregator dataframe index.
    :param scenario_types: List of selected scenario types.
    :return A dictionary of {aggregator planner_name: {aggregator scenario type: a list of (scenario type score,
    scenario log name)}}.
    """
    scenario_type_score_histogram_statistics_dict = _extract_metric_aggregator_scenario_type_score_data(
        metric_aggregator_dataframe=metric_aggregator_dataframe, scenario_types=scenario_types
    )
    # Get a list of histogram data
    histogram_data_list = extract_scenario_score_type_score_histogram_data(
        metric_aggregator_dataframe_index=metric_aggregator_dataframe_index,
        scenario_type_score_histogram_statistics_dict=scenario_type_score_histogram_statistics_dict,
        histogram_file_name=dataframe_file_name,
    )
    return histogram_data_list


def aggregate_metric_statistics_dataframe_histogram_data(
    metric_statistics_dataframe: MetricStatisticsDataFrame,
    metric_statistics_dataframe_index: int,
    metric_choices: List[str],
    scenario_types: Optional[Tuple[str]] = None,
) -> Optional[List[HistogramData]]:
    """
    Aggregate metric statistics dataframe data for histograms.
    :param metric_statistics_dataframe: Metric statistics dataframe.
    :param metric_statistics_dataframe_index: Metric statistics dataframe index.
    :param metric_choices: List of selected metrics.
    :param scenario_types: List of selected scenario types.
    :return A dictionary of statistics and their statistic data.
    """
    histogram_data_list = []
    if len(metric_choices) and metric_statistics_dataframe.metric_statistic_name not in metric_choices:
        return None

    planner_names = metric_statistics_dataframe.planner_names
    for planner_name in planner_names:
        histogram_statistics_dict = _extract_metric_statistics_dict(
            metric_statistics_dataframe=metric_statistics_dataframe,
            scenario_types=scenario_types,
            planner_name=planner_name,
        )
        if not histogram_statistics_dict:
            continue
        histogram_data_list.append(
            HistogramData(
                experiment_index=metric_statistics_dataframe_index,
                planner_name=planner_name,
                statistics=histogram_statistics_dict,
            )
        )
    return histogram_data_list


def compute_histogram_edges(
    bins: int, aggregated_data: Optional[HistogramConstantConfig.HistogramDataType]
) -> HistogramConstantConfig.HistogramEdgesDataType:
    """
    Compute histogram edges across different planners in the same metric statistics.
    :param bins: Number of bins.
    :param aggregated_data: Histogram aggregated data.
    :return Histogram edge data.
    """
    histogram_edge_data: HistogramConstantConfig.HistogramEdgesDataType = {}
    if aggregated_data is None:
        return histogram_edge_data

    for metric_statistics_name, aggregated_histogram_data in aggregated_data.items():
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
                _, edges = np.histogram(finite_values, bins=bins)
                histogram_edge_data[metric_statistics_name][statistic_name] = None if not len(edges) else edges

    return histogram_edge_data


def get_histogram_plot_x_range(unit: str, data: npt.NDArray[np.float64]) -> Union[List[str], FactorRange]:
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
