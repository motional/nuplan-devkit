from typing import Dict, List, Optional

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_violation import DrivableAreaViolationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_at_fault_collisions import EgoAtFaultCollisionStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_min_distance_to_lead_agent import EgoMinDistanceToLeadAgent
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision import TimeToCollisionStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoSafetyStatistics(MetricBase):
    """
    Ego safety performance metric. We assume that ego and other tracks do not drive in reverse mode (backwards).
    Checks if:
    1. Ego does not have an at_fault_collision, and
    2. Ego does not get too close to the front agent, and
    3. Ego maintains a minimum TTC greater than a given threahsold, and
    4. Ego drives in drivable area.
    """

    def __init__(
        self,
        name: str,
        category: str,
        time_to_collision_metric: TimeToCollisionStatistics,
        drivable_area_violation_metric: DrivableAreaViolationStatistics,
        ego_at_fault_collisions_metric: EgoAtFaultCollisionStatistics,
        ego_min_distance_to_lead_agent_metric: EgoMinDistanceToLeadAgent,
    ):
        """
        Initializes the EgoSafetyStatistics class
        :param name: Metric name
        :param category: Metric category
        :param time_to_collision_metric: time to collision metric
        :param drivable_area_violation_metric: drivable area violation metric
        :param ego_at_fault_collisions_metric: Ego at fault collisions metric
        :param ego_min_distance_to_lead_agent_metric: Minimum distance between ego and the front agent
        """
        super().__init__(name=name, category=category)

        self._time_to_collision = time_to_collision_metric
        self._drivable_area_violation = drivable_area_violation_metric
        self._at_fault_collisions = ego_at_fault_collisions_metric
        self._min_distance_to_lead_agent = ego_min_distance_to_lead_agent_metric

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        # Return 1.0 if safe, otherwise 0
        return float(metric_statistics[MetricStatisticsType.BOOLEAN].value)

    def check_ego_safety_performance(self, history: SimulationHistory, scenario: AbstractScenario) -> bool:
        """
        We assume that ego and other tracks do not drive in reverse mode (backwards).

        Returns True if:
        1. Ego does not have an at_fault_collision, and
        2. Ego does not get too close to the front agent, and
        3. Ego maintains a minimum TTC greater than a threahsold, and
        4. Ego drives in drivable area,
        Otherwise returns False

        :param history:  History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return True if safety performance is acceptable else False.
        """
        # Load pre-calculated violations from ego_at_fault_collision metric
        assert (
            self._at_fault_collisions.results
        ), "ego_at_fault_collisions metric must be run prior to calling {}".format(self.name)
        ego_at_fault_metric_count = self._at_fault_collisions.results[0].statistics[MetricStatisticsType.COUNT].value
        if ego_at_fault_metric_count > 0:
            return False

        # Load pre-calculated violations from ego_min_distance_to_lead_agent metric
        assert (
            self._min_distance_to_lead_agent.results
        ), "ego_min_distance_to_lead_agent metric must be run prior to calling {}".format(self.name)
        distance_to_lead_agents_within_bound = (
            self._min_distance_to_lead_agent.results[0].statistics[MetricStatisticsType.BOOLEAN].value
        )
        if not distance_to_lead_agents_within_bound:
            return False

        # Load pre-calculated TTC within bound from time_to_collision metric
        assert self._time_to_collision.results, "time_to_collision metric must be run prior to calling {}".format(
            self.name
        )
        time_to_collision_within_bound = (
            self._time_to_collision.results[0].statistics[MetricStatisticsType.BOOLEAN].value
        )
        if not time_to_collision_within_bound:
            return False

        # Load pre-calculated drivable area violation from drivable_area_violation metric
        assert (
            self._drivable_area_violation.results
        ), "drivable_area_violation metric must be run prior to calling {}".format(self.name)
        number_of_drivable_area_violation = (
            self._drivable_area_violation.results[0].statistics[MetricStatisticsType.COUNT].value
        )
        if number_of_drivable_area_violation > 0:
            return False

        return True

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        safety_performance_metric = self.check_ego_safety_performance(history=history, scenario=scenario)
        statistics = {
            MetricStatisticsType.BOOLEAN: Statistic(
                name="ego_safety_performance", unit="boolean", value=safety_performance_metric
            )
        }

        results = self._construct_metric_results(metric_statistics=statistics, time_series=None, scenario=scenario)
        return results  # type: ignore
