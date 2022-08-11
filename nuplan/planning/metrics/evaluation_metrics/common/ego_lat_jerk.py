from typing import List

from nuplan.planning.metrics.evaluation_metrics.base.within_bound_metric_base import WithinBoundMetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics
from nuplan.planning.metrics.utils.state_extractors import extract_ego_jerk
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoLatJerkStatistics(WithinBoundMetricBase):
    """Ego lateral jerk metric."""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializes the EgoLatJerkStatistics class
        :param name: Metric name
        :param category: Metric category.
        """
        super().__init__(name=name, category=category)

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the lateral jerk  metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated lateral jerk metric.
        """
        metric_statistics: List[MetricStatistics] = self._compute_statistics(
            history=history,
            scenario=scenario,
            statistic_unit_name='meters_per_second_cubed',
            extract_function=extract_ego_jerk,
            extract_function_params={'acceleration_coordinate': 'y'},
        )
        return metric_statistics
