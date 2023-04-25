from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import pandas
import pyarrow
import pyarrow.parquet as pq

from nuplan.common.utils.io_utils import safe_path_to_string
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame


class AbstractMetricAggregator(metaclass=ABCMeta):
    """Interface for metric aggregator"""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the metric aggregator name
        :return the metric aggregator name.
        """
        pass

    @property
    @abstractmethod
    def final_metric_score(self) -> Optional[float]:
        """Returns the final metric score."""
        pass

    @abstractmethod
    def __call__(self, metric_dataframes: Dict[str, MetricStatisticsDataFrame]) -> None:
        """
        Run an aggregator to generate an aggregated parquet file
        :param metric_dataframes: A dictionary of metric name and dataframe.
        """
        pass

    @staticmethod
    def _save_with_metadata(dataframe: pandas.DataFrame, save_path: Path, metadata: Dict[str, str]) -> None:
        """
        Save to a parquet file with additional metadata using pyarrow
        :param dataframe: Pandas dataframe
        :param save_path: Path to save the dataframe.
        """
        pyarrow_table = pyarrow.Table.from_pandas(df=dataframe)
        schema_metadata = pyarrow_table.schema.metadata
        schema_metadata.update(metadata)
        updated_schema = pyarrow_table.schema.with_metadata(schema_metadata)
        pyarrow_table = pyarrow_table.cast(updated_schema)
        pq.write_table(pyarrow_table, str(save_path))

    @staticmethod
    def _save_parquet(dataframe: pandas.DataFrame, save_path: Path) -> None:
        """
        Save dataframe to a parquet file.
        The path can be local or s3.
        :param dataframe: Pandas dataframe.
        :param save_path: Path to save the dataframe.
        """
        dataframe.to_parquet(safe_path_to_string(save_path))

    @abstractmethod
    def read_parquet(self) -> None:
        """Read a parquet file, and update the dataframe."""
        pass

    @property
    @abstractmethod
    def parquet_file(self) -> Path:
        """Getter for the path to the generated parquet file."""
        pass

    @property
    @abstractmethod
    def challenge(self) -> Optional[str]:
        """Returns the name of the challenge, if applicable."""
        pass
