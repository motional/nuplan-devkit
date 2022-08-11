from abc import ABCMeta, abstractmethod
from typing import List, Optional

from nuplan.planning.metrics.metric_result import MetricStatistics, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class AbstractMetricBuilder(metaclass=ABCMeta):
    """Interface for generic metric."""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the metric name
        :return the metric name.
        """
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """
        Returns the metric category
        :return the metric category.
        """
        pass

    @abstractmethod
    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """
        Compute a final score from this metric
        :param scenario: Scenario running this metric
        :param metric_statistics: A dictionary of statistics
        :param time_series: Time series
        :return A metric cost score.
        """
        pass

    @abstractmethod
    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        pass
