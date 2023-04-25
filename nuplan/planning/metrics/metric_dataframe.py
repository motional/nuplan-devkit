from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas

from nuplan.common.utils.io_utils import safe_path_to_string


@dataclass
class MetricStatisticsDataFrame:
    """Metric statistics data frame class."""

    metric_statistic_name: str
    metric_statistics_dataframe: pandas.DataFrame
    time_series_unit_column: ClassVar[str] = 'time_series_unit'
    time_series_timestamp_column: ClassVar[str] = 'time_series_timestamps'
    time_series_values_column: ClassVar[str] = 'time_series_values'
    time_series_selected_frames_column: ClassVar[str] = 'time_series_selected_frames'

    def __eq__(self, other: object) -> bool:
        """Compare equality."""
        if not isinstance(other, MetricStatisticsDataFrame):
            return NotImplemented

        return self.metric_statistic_name == other.metric_statistic_name and self.metric_statistics_dataframe.equals(
            other.metric_statistics_dataframe
        )

    def __hash__(self) -> int:
        """Implement hash for caching."""
        return hash(self.metric_statistic_name) + id(self.metric_statistics_dataframe)

    @classmethod
    def load_parquet(cls, parquet_path: Path) -> MetricStatisticsDataFrame:
        """
        Load a parquet file to this class.
        The path can be local or s3.
        :param parquet_path: A path to a parquet file.
        """
        data_frame = pandas.read_parquet(path=safe_path_to_string(parquet_path))
        try:
            # Check if any rows.
            if not len(data_frame):
                raise IndexError

            # Load the main name of the metric statistics name by reading the first row.
            metric_statistics_name = data_frame['metric_statistics_name'][0]
        except (IndexError, Exception):
            # if failed to load the file, then we Assume the parquet file name is the metric statistic name
            metric_statistics_name = parquet_path.stem

        return MetricStatisticsDataFrame(
            metric_statistic_name=metric_statistics_name, metric_statistics_dataframe=data_frame
        )

    @lru_cache
    def query_scenarios(
        self,
        scenario_names: Optional[Tuple[str]] = None,
        scenario_types: Optional[Tuple[str]] = None,
        planner_names: Optional[Tuple[str]] = None,
        log_names: Optional[Tuple[str]] = None,
    ) -> pandas.DataFrame:
        """
        Query scenarios with a list of scenario types and planner names.
        :param scenario_names: A tuple of scenario names.
        :param scenario_types: A tuple of scenario types.
        :param planner_names: A tuple of planner names.
        :param log_names: A tuple of log names.
        :return Pandas dataframe after filtering.
        """
        if not scenario_names and not scenario_types and not planner_names:
            return self.metric_statistics_dataframe

        default_query: npt.NDArray[np.bool_] = np.asarray([True] * len(self.metric_statistics_dataframe.index))
        scenario_name_query = (
            self.metric_statistics_dataframe['scenario_name'].isin(scenario_names) if scenario_names else default_query
        )

        scenario_type_query = (
            self.metric_statistics_dataframe['scenario_type'].isin(scenario_types) if scenario_types else default_query
        )

        planner_name_query = (
            self.metric_statistics_dataframe['planner_name'].isin(planner_names) if planner_names else default_query
        )

        log_name_query = self.metric_statistics_dataframe['log_name'].isin(log_names) if log_names else default_query

        return self.metric_statistics_dataframe[
            scenario_name_query & scenario_type_query & planner_name_query & log_name_query
        ]

    @cached_property
    def metric_statistics_names(self) -> List[str]:
        """Return metric statistic names."""
        return list(self.metric_statistics_dataframe['metric_statistics_name'].unique())

    @cached_property
    def metric_computator(self) -> str:
        """Return metric computator."""
        if len(self.metric_statistics_dataframe):
            return self.metric_statistics_dataframe["metric_computator"][0]  # type: ignore
        else:
            raise IndexError("No available records found!")

    @cached_property
    def metric_category(self) -> str:
        """Return metric category."""
        if len(self.metric_statistics_dataframe):
            return self.metric_statistics_dataframe["metric_category"][0]  # type: ignore
        else:
            raise IndexError("No available records found!")

    @cached_property
    def metric_score_unit(self) -> str:
        """Return metric score unit."""
        return self.metric_statistics_dataframe["metric_score_unit"][0]  # type: ignore

    @cached_property
    def scenario_types(self) -> List[str]:
        """Return a list of scenario types."""
        return list(self.metric_statistics_dataframe["scenario_type"].unique())

    @cached_property
    def scenario_names(self) -> List[str]:
        """Return a list of scenario names."""
        return list(self.metric_statistics_dataframe["scenario_name"])

    @cached_property
    def column_names(self) -> List[str]:
        """Return a list of column names in a table."""
        return list(self.metric_statistics_dataframe.columns)

    @cached_property
    def statistic_names(self) -> List[str]:
        """Return a list of statistic names in a table."""
        return [col.split('_stat_type')[0] for col in self.column_names if '_stat_type' in col]

    @cached_property
    def time_series_headers(self) -> List[str]:
        """Return time series headers."""
        return [self.time_series_unit_column, self.time_series_timestamp_column, self.time_series_values_column]

    @cached_property
    def get_time_series_selected_frames(self) -> Optional[List[int]]:
        """Return selected frames in time series."""
        try:
            return self.metric_statistics_dataframe[self.time_series_selected_frames_column].iloc[0]  # type: ignore
        except KeyError:
            return None

    @cached_property
    def time_series_dataframe(self) -> pandas.DataFrame:
        """Return time series dataframe."""
        return self.metric_statistics_dataframe.loc[:, self.time_series_headers]

    @lru_cache
    def statistics_dataframe(self, statistic_names: Optional[Tuple[str]] = None) -> pandas.DataFrame:
        """
        Return statistics columns
        :param statistic_names: A list of statistic names to query
        :return Pandas dataframe after querying.
        """
        if statistic_names:
            return self.metric_statistics_dataframe[statistic_names]

        statistic_headers = []
        for column_name in self.column_names:
            for statistic_name in self.statistic_names:
                if statistic_name in column_name:
                    statistic_headers.append(column_name)
                    continue

        return self.metric_statistics_dataframe[statistic_headers]

    @cached_property
    def planner_names(self) -> List[str]:
        """Return a list of planner names."""
        return list(self.metric_statistics_dataframe['planner_name'].unique())
