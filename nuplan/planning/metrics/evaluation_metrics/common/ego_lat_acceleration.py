from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoLatAccelerationStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Ego lateral acceleration metric.
        :param name: Metric name.
        :param category: Metric category.
        """

        self._name = name
        self._category = category

    @property
    def name(self) -> str:
        """
        Returns the metric name.
        :return: the metric name.
        """

        return self._name

    @property
    def category(self) -> str:
        """
        Returns the metric category.
        :return: the metric category.
        """

        return self._category

    @staticmethod
    def ego_lat_accel(history: SimulationHistory) -> npt.NDArray[np.float32]:
        """
        Compute ego acceleration in longitudinal.
        :param history: History from a simulation engine.
        :return An array of longitudinal accelerations in [N].
        """

        # Ego velocities are defined in ego's local frame
        lat_accels = np.asarray([sample.ego_state.dynamic_car_state.center_acceleration_2d.y
                                 for sample in history.data])

        return lat_accels

    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """

        lat_accels = self.ego_lat_accel(history=history)
        statistics = {
            MetricStatisticsType.MAX: Statistic(name="ego_max_lat_acceleration", unit="meters_per_second_squared",
                                                value=np.amax(lat_accels)),
            MetricStatisticsType.MIN: Statistic(name="ego_min_lat_acceleration", unit="meters_per_second_squared",
                                                value=np.amin(lat_accels)),
            MetricStatisticsType.P90: Statistic(name="ego_p90_lat_acceleration", unit="meters_per_second_squared",
                                                value=np.percentile(np.abs(lat_accels), 90)),  # type:ignore
        }

        time_stamps = extract_ego_time_point(history)
        time_series = TimeSeries(unit='meters_per_second_squared',
                                 time_stamps=list(time_stamps),
                                 values=list(lat_accels))
        result = MetricStatistics(metric_computator=self.name,
                                  name="ego_lat_acceleration_statistics",
                                  statistics=statistics,
                                  time_series=time_series,
                                  metric_category=self.category)

        return [result]
