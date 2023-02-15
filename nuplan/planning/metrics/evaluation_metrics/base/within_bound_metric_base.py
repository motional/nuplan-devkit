from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class WithinBoundMetricBase(MetricBase):
    """Base class for evaluation of within_bound metrics."""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializes the WithinBoundMetricBase class
        :param name: Metric name
        :param category: Metric category.
        """
        super().__init__(name=name, category=category)
        self.within_bound_status: Optional[bool] = False

    @staticmethod
    def _compute_within_bound_metric_score(within_bound_status: bool) -> float:
        """
        Compute a metric score based on within bound condition
        :param within_bound_status: True if the value is within the bound, otherwise false
        :return 1.0 if within_bound_status is true otherwise 0.
        """
        return 1.0 if within_bound_status else 0.0

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> Optional[float]:
        """Inherited, see superclass."""
        return None

    @staticmethod
    def _compute_within_bound(
        time_series: TimeSeries,
        min_within_bound_threshold: Optional[float] = None,
        max_within_bound_threshold: Optional[float] = None,
    ) -> Optional[bool]:
        """
        Compute if value is within bound based on the thresholds
        :param time_series: Time series object
        :param min_within_bound_threshold: Minimum threshold to check if value is within bound
        :param max_within_bound_threshold: Maximum threshold to check if value is within bound.
        """
        ego_pose_values: npt.NDArray[np.float32] = np.array(time_series.values)
        if not min_within_bound_threshold and not max_within_bound_threshold:
            return None

        # Set to negative infinity
        if min_within_bound_threshold is None:
            min_within_bound_threshold = float(-np.inf)

        # Set to positive infinity
        if max_within_bound_threshold is None:
            max_within_bound_threshold = float(np.inf)

        ego_pose_value_within_bound = (ego_pose_values > min_within_bound_threshold) & (
            ego_pose_values < max_within_bound_threshold
        )

        # Return true if all abs values within the bound
        return bool(np.all(ego_pose_value_within_bound))

    def _compute_statistics(
        self,
        history: SimulationHistory,
        scenario: AbstractScenario,
        statistic_unit_name: str,
        extract_function: Any,
        extract_function_params: Dict[str, Any],
        min_within_bound_threshold: Optional[float] = None,
        max_within_bound_threshold: Optional[float] = None,
    ) -> List[MetricStatistics]:
        """
        Compute metrics following the same structure
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :param statistic_unit_name: Statistic unit name
        :param extract_function: Function used to extract certain values
        :param extract_function_params: Params used in extract_function
        :param min_within_bound_threshold: Minimum threshold to check if value is within bound
        :param max_within_bound_threshold: Maximum threshold to check if value is within bound.
        """
        ego_pose_states = history.extract_ego_state

        # Extract attribute
        ego_pose_values = extract_function(ego_pose_states, **extract_function_params)

        # Extract timestamps
        ego_pose_timestamps = extract_ego_time_point(ego_pose_states)

        # Timestamps
        time_series = TimeSeries(
            unit=statistic_unit_name, time_stamps=list(ego_pose_timestamps), values=list(ego_pose_values)
        )

        # Compute statistics
        statistics_type_list = [
            MetricStatisticsType.MAX,
            MetricStatisticsType.MIN,
            MetricStatisticsType.MEAN,
            MetricStatisticsType.P90,
        ]

        metric_statistics = self._compute_time_series_statistic(
            time_series=time_series, statistics_type_list=statistics_type_list
        )

        self.within_bound_status = self._compute_within_bound(
            time_series=time_series,
            min_within_bound_threshold=min_within_bound_threshold,
            max_within_bound_threshold=max_within_bound_threshold,
        )
        if self.within_bound_status is not None:
            metric_statistics.append(
                Statistic(
                    name=f'abs_{self.name}_within_bounds',
                    unit=MetricStatisticsType.BOOLEAN.unit,
                    value=self.within_bound_status,
                    type=MetricStatisticsType.BOOLEAN,
                )
            )

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=metric_statistics, time_series=time_series, scenario=scenario
        )

        return results

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return: the estimated metric.
        """
        # Subclasses should implement this
        raise NotImplementedError
