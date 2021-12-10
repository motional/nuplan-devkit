from abc import ABCMeta, abstractmethod
from typing import List

from nuplan.planning.metrics.metric_result import MetricStatistics
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class AbstractMetricBuilder(metaclass=ABCMeta):
    """
    Interface for generic metric
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the metric name.
        :return: the metric name.
        """
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """
        Returns the metric category.
        :return: the metric category.
        """
        pass

    @abstractmethod
    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """

        pass
