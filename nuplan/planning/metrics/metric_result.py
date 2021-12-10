from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricStatisticsType(Enum):
    MAX: str = 'MAX'
    MIN: str = 'MIN'
    P90: str = 'P90'
    MEAN: str = 'MEAN'
    COUNT: str = 'COUNT'
    VALUE: str = 'VALUE'
    DISTANCE: str = 'DISTANCE'
    VELOCITY: str = 'VELOCITY'
    BOOLEAN: str = 'BOOLEAN'

    def __repr__(self) -> str:
        """ Metric type string representation. """

        return self.value

    def serialize(self) -> str:
        """ Serialize the type when saving. """

        return self.value

    @classmethod
    def deserialize(cls, key: str) -> MetricStatisticsType:
        """ Deserialize the type when loading from a string. """

        return MetricStatisticsType.__members__[key]


@dataclass
class MetricResult(ABC):

    metric_computator: str  # Name of metric computator
    name: str  # Name of the metric
    metric_category: str  # Category of metric

    def serialize(self) -> Dict[str, Any]:
        """ Serialize the metric result. """

        pass

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MetricResult:
        """
        Deserialize the metric result when loading from a file.
        :param data; A dictionary of data in loading.
        """

        pass


@dataclass
class Statistic:
    name: str  # name of statistic
    unit: str  # unit of statistic
    value: float  # value of the statistic

    def serialize(self) -> Dict[str, Any]:
        """ Serialization of TimeSeries. """

        return {'name': self.name,
                'unit': self.unit,
                'value': self.value}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Statistic:
        """ Deserialization of TimeSeries.
        :param data: A dictionary of data,
        :return A Statistic data class.
        """

        return Statistic(name=data['name'], unit=data['unit'], value=data['value'])


@dataclass
class TimeSeries:
    unit: str  # unit of the time series
    time_stamps: List[int]  # time stamps of the time series [microseconds]
    values: List[float]  # values of the time series

    def __post_init__(self) -> None:
        assert len(self.time_stamps) == len(self.values)

    def serialize(self) -> Dict[str, Any]:
        """ Serialization of TimeSeries. """

        return {'unit': self.unit,
                'time_stamps': self.time_stamps,
                'values': self.values}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Optional[TimeSeries]:
        """ Deserialization of TimeSeries.
        :param data: A dictionary of data,
        :return A TimeSeries dataclass.
        """

        return TimeSeries(unit=data['unit'],
                          time_stamps=data['time_stamps'],
                          values=data['values']) if data is not None else None


@dataclass
class MetricStatistics(MetricResult):

    statistics: Dict[MetricStatisticsType, Statistic]  # Collection of statistics
    time_series: Optional[TimeSeries]  # time series data of the metric

    def serialize(self) -> Dict[str, Any]:
        """ Serialize the metric result. """

        return {'metric_computator': self.metric_computator,
                'name': self.name,
                'statistics': {statistic_type.serialize(): statistics.serialize()
                               for statistic_type, statistics in self.statistics.items()},
                'time_series': self.time_series.serialize() if self.time_series is not None else None,
                'metric_category': self.metric_category
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
            statistics={MetricStatisticsType.deserialize(statistic_type): Statistic.deserialize(statistics)
                        for statistic_type, statistics in data['statistics'].items()},
            time_series=TimeSeries.deserialize(data['time_series']),
            metric_category=data['metric_category']
        )


@dataclass
class MetricViolation(MetricResult):
    unit: str  # unit of the violation
    start_timestamp: int  # start time stamp of the violation [microseconds]
    duration: int  # duration of the violation [microseconds]
    extremum: float  # the most violating value of the violation
    mean: float  # The average violation level

    def serialize(self) -> Dict[str, Any]:
        """ Serialize the metric result. """

        return {'metric_computator': self.metric_computator,
                'name': self.name,
                'unit': self.unit,
                'start_timestamp': self.start_timestamp,
                'duration': self.duration,
                'extremum': self.extremum,
                'metric_category': self.metric_category
                }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MetricViolation:
        """
        Deserialize the metric result when loading from a file.
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
            mean=data['mean']
        )
