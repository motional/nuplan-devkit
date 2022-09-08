from typing import List, Optional

import numpy as np

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_jerk import EgoJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_acceleration import EgoLatAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration import EgoLonAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_jerk import EgoLonJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_acceleration import EgoYawAccelerationStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_rate import EgoYawRateStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoIsComfortableStatistics(MetricBase):
    """
    Check if ego trajectory is comfortable based on min_ego_lon_acceleration, max_ego_lon_acceleration,
    max_ego_abs_lat_acceleration, max_ego_abs_yaw_rate, max_ego_abs_yaw_acceleration, max_ego_abs_jerk_lon,
    max_ego_abs_jerk.
    """

    def __init__(
        self,
        name: str,
        category: str,
        ego_jerk_metric: EgoJerkStatistics,
        ego_lat_acceleration_metric: EgoLatAccelerationStatistics,
        ego_lon_acceleration_metric: EgoLonAccelerationStatistics,
        ego_lon_jerk_metric: EgoLonJerkStatistics,
        ego_yaw_acceleration_metric: EgoYawAccelerationStatistics,
        ego_yaw_rate_metric: EgoYawRateStatistics,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initializes the EgoIsComfortableStatistics class
        :param name: Metric name
        :param category: Metric category
        :param ego_jerk_metric: Ego jerk metric
        :param ego_lat_acceleration_metric: Ego lat acceleration metric
        :param ego_lon_acceleration_metric: Ego lon acceleration metric
        :param ego_lon_jerk_metric: Ego lon jerk metric
        :param ego_yaw_acceleration_metric: Ego yaw acceleration metric
        :param ego_yaw_rate_metric: Ego yaw rate metric.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._comfortability_metrics = [
            ego_jerk_metric,
            ego_lat_acceleration_metric,
            ego_lon_acceleration_metric,
            ego_lon_jerk_metric,
            ego_yaw_acceleration_metric,
            ego_yaw_rate_metric,
        ]

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)

    def check_ego_is_comfortable(self, history: SimulationHistory, scenario: AbstractScenario) -> bool:
        """
        Check if ego trajectory is comfortable
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return Ego comfortable status.
        """
        metrics_results = [metric.within_bound_status for metric in self._comfortability_metrics]
        ego_is_comfortable = bool(np.all(metrics_results))

        return ego_is_comfortable

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        ego_is_comfortable = self.check_ego_is_comfortable(history=history, scenario=scenario)
        statistics = [
            Statistic(
                name='ego_is_comfortable',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=ego_is_comfortable,
                type=MetricStatisticsType.BOOLEAN,
            )
        ]

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit
        )
        return results
