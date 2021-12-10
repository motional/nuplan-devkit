from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from nuplan.planning.metrics.metric_result import MetricStatistics


@dataclass
class MetricFileKey:

    metric_name: str
    scenario_name: str
    scenario_type: str
    planner_name: str

    def serialize(self) -> Dict[str, str]:
        """ Serialization of metric result key. """

        return {'metric_name': self.metric_name,
                'scenario_name': self.scenario_name,
                'scenario_type': self.scenario_type,
                'planner_name': self.planner_name}

    @classmethod
    def deserialize(cls, data: Dict[str, str]) -> MetricFileKey:
        """ Deserialization of .
        :param data: A dictionary of data,
        :return A Statistic data class.
        """

        return MetricFileKey(metric_name=data['metric_name'],
                             scenario_name=data['scenario_name'],
                             scenario_type=data['scenario_type'],
                             planner_name=data['planner_name'])


@dataclass
class MetricFile:
    """ Metric storage result. """

    key: MetricFileKey  # Metric file key

    # {metric statistics name: # a list of metric statistics}
    metric_statistics: Dict[str, List[MetricStatistics]] = field(default_factory=dict)

    def serialize(self) -> Dict[str, Any]:
        """ Serialization of metric result key. """

        return {
            'key': self.key.serialize(),
            'metric_statistics': {statistic_name: [metric_statistic.serialize() for metric_statistic
                                                   in metric_statistics] for statistic_name, metric_statistics
                                  in self.metric_statistics.items()}
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MetricFile:
        """
        Deserialization of metric storage result.
        :param data: A dictionary of data,
        :return A Statistic data class.
        """

        metric_statistics = {
            statistic_name: [MetricStatistics.deserialize(statistic) for statistic in statistics]
            for statistic_name, statistics in data['metric_statistics'].items()
        }
        metric_file_key = MetricFileKey.deserialize(data['key'])
        return MetricFile(key=metric_file_key,
                          metric_statistics=metric_statistics)
