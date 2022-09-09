from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame


class MetricStatisticsType(Enum):
    """Enum of different types for statistics."""

    MAX = 'MAX'
    MIN = 'MIN'
    P90 = 'P90'
    MEAN = 'MEAN'
    VALUE = 'VALUE'
    VELOCITY = 'VELOCITY'
    BOOLEAN = 'BOOLEAN'
    RATIO = 'RATIO'
    COUNT = 'COUNT'

    def __str__(self) -> str:
        """Metric type string representation."""
        return str(self.value)

    def __repr__(self) -> str:
        """Metric type string representation."""
        return str(self.value)

    @property
    def unit(self) -> str:
        """Get a default unit with a type."""
        if self.value == 'BOOLEAN':
            return 'boolean'
        elif self.value == 'RATIO':
            return 'ratio'
        elif self.value == 'COUNT':
            return 'count'
        else:
            raise ValueError(f"{self.value} don't have a default unit!")

    def serialize(self) -> str:
        """Serialize the type when saving."""
        return self.value

    @classmethod
    def deserialize(cls, key: str) -> MetricStatisticsType:
        """Deserialize the type when loading from a string."""
        return MetricStatisticsType.__members__[key]


@dataclass
class MetricResult(ABC):
    """
    Abstract MetricResult.
    """

    metric_computator: str  # Name of metric computator
    name: str  # Name of the metric
    metric_category: str  # Category of metric

    def serialize(self) -> Dict[str, Any]:
        """Serialize the metric result."""
        pass

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MetricResult:
        """
        Deserialize the metric result when loading from a file.
        :param data; A dictionary of data in loading.
        """
        pass

    def serialize_dataframe(self) -> Dict[str, Any]:
        """
        Serialize a dictionary for dataframe.
        :return a dictionary
        """
        pass


@dataclass
class Statistic:
    """
    Class to report statsitcs of metrics.
    """

    name: str  # name of statistic
    unit: str  # unit of statistic
    type: MetricStatisticsType  # Metric statistic type
    value: Union[float, bool]  # value of the statistic

    def serialize(self) -> Dict[str, Any]:
        """Serialization of TimeSeries."""
        return {'name': self.name, 'unit': self.unit, 'value': self.value, 'type': self.type.serialize()}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Statistic:
        """
        Deserialization of TimeSeries
        :param data: A dictionary of data
        :return A Statistic data class.
        """
        return Statistic(
            name=data['name'],
            unit=data['unit'],
            value=data['value'],
            type=MetricStatisticsType.deserialize(data['type']),
        )


@dataclass
class TimeSeries:
    """
    Class to report time series data of metrics.
    """

    unit: str  # unit of the time series
    time_stamps: List[int]  # time stamps of the time series [microseconds]
    values: List[float]  # values of the time series
    selected_frames: Optional[List[int]] = None  # Selected frames

    def __post_init__(self) -> None:
        """Post initialization of TimeSeries."""
        assert len(self.time_stamps) == len(self.values)

    def serialize(self) -> Dict[str, Any]:
        """Serialization of TimeSeries."""
        return {
            'unit': self.unit,
            'time_stamps': self.time_stamps,
            'values': self.values,
            'selected_frames': self.selected_frames,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Optional[TimeSeries]:
        """
        Deserialization of TimeSeries
        :param data: A dictionary of data
        :return A TimeSeries dataclass.
        """
        return (
            TimeSeries(
                unit=data['unit'],
                time_stamps=data['time_stamps'],
                values=data['values'],
                selected_frames=data['selected_frames'],
            )
            if data is not None
            else None
        )


@dataclass
class MetricStatistics(MetricResult):
    """Class to report results of metric statistics."""

    statistics: List[Statistic]  # Collection of statistics
    time_series: Optional[TimeSeries] = None  # Time series data of the metric
    metric_score: Optional[float] = None  # Final score of a metric in a scenario
    metric_score_unit: Optional[str] = None  # Final score unit, for example, float or bool

    def serialize(self) -> Dict[str, Any]:
        """Serialize the metric result."""
        return {
            'metric_computator': self.metric_computator,
            'name': self.name,
            'statistics': [statistic.serialize() for statistic in self.statistics],
            'time_series': self.time_series.serialize() if self.time_series is not None else None,
            'metric_category': self.metric_category,
            'metric_score': self.metric_score,
            'metric_score_unit': self.metric_score_unit,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MetricStatistics:
        """
        Deserialize the metric result when loading from a file.
        :param data; A dictionary of data in loading.
        """
        return MetricStatistics(
            metric_computator=data['metric_computator'],
            name=data['name'],
            statistics=[Statistic.deserialize(statistic) for statistic in data['statistics']],
            time_series=TimeSeries.deserialize(data['time_series']),
            metric_category=data['metric_category'],
            metric_score=data['metric_score'],
            metric_score_unit=data['metric_score_unit'],
        )

    def serialize_dataframe(self) -> Dict[str, Any]:
        """
        Serialize a dictionary for dataframe
        :return a dictionary
        """
        columns: Dict[str, Any] = {
            'metric_score': self.metric_score,
            'metric_score_unit': self.metric_score_unit,
            'metric_category': self.metric_category,
        }
        for statistic in self.statistics:
            statistic_columns = {
                f'{statistic.name}_stat_type': statistic.type.serialize(),
                f'{statistic.name}_stat_unit': [statistic.unit],
                f'{statistic.name}_stat_value': [statistic.value],
            }
            columns.update(statistic_columns)

        time_series_columns: Dict[str, List[Any]] = {}
        if self.time_series is None:
            time_series_columns.update(
                {
                    MetricStatisticsDataFrame.time_series_unit_column: [None],
                    MetricStatisticsDataFrame.time_series_timestamp_column: [None],
                    MetricStatisticsDataFrame.time_series_values_column: [None],
                    MetricStatisticsDataFrame.time_series_selected_frames_column: [None],
                }
            )
        else:
            time_series_columns.update(
                {
                    MetricStatisticsDataFrame.time_series_unit_column: [self.time_series.unit],
                    MetricStatisticsDataFrame.time_series_timestamp_column: [
                        [int(timestamp) for timestamp in self.time_series.time_stamps]
                    ],
                    MetricStatisticsDataFrame.time_series_values_column: [self.time_series.values],
                    MetricStatisticsDataFrame.time_series_selected_frames_column: [self.time_series.selected_frames],
                }
            )

        columns.update(time_series_columns)
        return columns


@dataclass
class MetricViolation(MetricResult):
    """Class to report results of violation-based metrics."""

    unit: str  # unit of the violation
    start_timestamp: int  # start time stamp of the violation [microseconds]
    duration: int  # duration of the violation [microseconds]
    extremum: float  # the most violating value of the violation
    mean: float  # The average violation level

    def serialize(self) -> Dict[str, Any]:
        """Serialize the metric result."""
        return {
            'metric_computator': self.metric_computator,
            'name': self.name,
            'unit': self.unit,
            'start_timestamp': self.start_timestamp,
            'duration': self.duration,
            'extremum': self.extremum,
            'metric_category': self.metric_category,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MetricViolation:
        """
        Deserialize the metric result when loading from a file
        :param data; A dictionary of data in loading.
        """
        return MetricViolation(
            metric_computator=data['metric_computator'],
            name=data['name'],
            start_timestamp=data['start_timestamp'],
            duration=data['duration'],
            extremum=data['extremum'],
            unit=data['unit'],
            metric_category=data['metric_category'],
            mean=data['mean'],
        )
