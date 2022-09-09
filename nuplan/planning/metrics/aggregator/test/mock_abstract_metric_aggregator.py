from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas

from nuplan.planning.metrics.aggregator.abstract_metric_aggregator import AbstractMetricAggregator
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame

logger = logging.getLogger(__name__)


class MockAbstractMetricAggregator(AbstractMetricAggregator):
    """Mock Metric aggregator."""

    def __init__(
        self,
        aggregator_save_path: Path,
        name: str = 'dummy_metric_aggregator',
        metric_weights: Optional[Dict[str, float]] = None,
        file_name: str = 'dummy_metric_aggregator.parquet',
    ):
        """
        Initializer for MockAbstractMetricAggregator class
        :param name: Metric aggregator name
        :param metric_weights: Weights for each metric. Default would be 1.0
        :param file_name: Saved file name
        :param aggregator_save_path: Save path for this aggregated parquet file.
        """
        self._name = name
        self._metric_weights = metric_weights or {'default': 1.0}
        self._file_name = file_name
        self._aggregator_save_path = aggregator_save_path
        if not self._aggregator_save_path.exists():
            self._aggregator_save_path.mkdir(exist_ok=True, parents=True)

        self._parquet_file = self._aggregator_save_path / self._file_name
        self._aggregated_metric_dataframe: Optional[pandas.DataFrame] = None

    @property
    def aggregated_metric_dataframe(self) -> Optional[pandas.DataFrame]:
        """Return the aggregated metric dataframe."""
        return self._aggregated_metric_dataframe

    @property
    def name(self) -> str:
        """
        Return the metric aggregator name
        :return: the metric aggregator name.
        """
        return self._name

    @property
    def final_metric_score(self) -> Optional[float]:
        """Return the final metric score."""
        if self._aggregated_metric_dataframe is not None:
            return self._aggregated_metric_dataframe.iloc[-1, -1]  # type: ignore
        else:
            logger.warning("The metric not yet aggregated.")
            return None

    def _get_metric_weight(self, metric_name: str) -> float:
        """
        Get metric weights
        :param metric_name: The metric name
        :return: Weight for the metric.
        """
        metric_weight = self._metric_weights.get(metric_name, None)
        if not metric_weight:
            metric_weight = self._metric_weights.get('default', 1.0)

        return metric_weight  # type: ignore

    def __call__(self, metric_dataframes: Dict[str, MetricStatisticsDataFrame]) -> None:
        """
        Run an aggregator to generate an aggregated parquet file
        :param metric_dataframes: A dictionary of metric name and dataframe.
        """
        # Create dummy dataframe

        dataframe_columns = {"test_column_1": [1, 2, 3], "test_column_2": [4, 5, 6]}
        # Convert to pandas dataframe
        self._aggregated_metric_dataframe = pandas.DataFrame(data=dataframe_columns)

        # Save to a parquet file
        self._save_parquet(dataframe=self._aggregated_metric_dataframe, save_path=self._parquet_file)

    def read_parquet(self) -> None:
        """Read a parquet file."""
        self._aggregated_metric_dataframe = pandas.read_parquet(self._parquet_file)

    @property
    def parquet_file(self) -> Path:
        """Inherited, see superclass."""
        return self._parquet_file

    @property
    def challenge(self) -> Optional[str]:
        """Inherited, see superclass."""
        return None
