from typing import List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import box3d_to_polygon, ego_state_to_box_3d
from shapely.geometry import Polygon


class EgoMinDistanceToTrackStatistics(AbstractMetricBuilder):

    def __init__(self, name: str, category: str) -> None:
        """
        Ego minimum distance to any tracks metric.
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
    def extract_tracks(history_samples: List[SimulationHistorySample]) -> List[Optional[List[Polygon]]]:
        """
        Extract tracks based on detections.
        :param history_samples: A list of history samples in scenario.
        :return A list of arrays of box/track centers.
        """

        tracks: List[Optional[List[Polygon]]] = []
        for history_sample in history_samples:
            detections = history_sample.observation
            if not isinstance(detections, Detections):
                tracks.append(None)
            else:
                boxes = detections.boxes
                track_polygons = [box3d_to_polygon(box) for box in boxes]
                tracks.append(track_polygons)

        return tracks

    def compute_ego_track_min_distance(self, history: SimulationHistory) -> npt.NDArray[np.float32]:
        """
        Compute minimum distance of ego poses and tracks (detections) in a scenario.
        :param history: History from a simulation engine.
        :return: An array of absolute minimum distances between ego poses and tracks in a scenario.
        """

        # Extract trajectory of each box.
        scenario_tracks = self.extract_tracks(history.data)

        # Extract ego pose center x and y.
        ego_pose_polygons = [box3d_to_polygon(ego_state_to_box_3d(sample.ego_state)) for sample in history.data]

        min_distances = []
        assert len(scenario_tracks) == len(ego_pose_polygons), "Length of tracks and ego poses should be the same."

        for ego_poses_polygon, track_polygons in zip(ego_pose_polygons, scenario_tracks):
            if not track_polygons:
                min_distances.append(np.inf)
            else:
                min_distance = np.inf

                # Distance is zero if one is within another polygon.
                for track_polygon in track_polygons:
                    distance = ego_poses_polygon.distance(track_polygon)
                    if distance < min_distance:
                        min_distance = distance
                min_distances.append(min_distance)

        return np.asarray(min_distances)

    def compute(self, history: SimulationHistory) -> List[MetricStatistics]:
        """
        Returns the estimated metric.
        :param history: History from a simulation engine.
        :return: the estimated metric.
        """

        min_distances = self.compute_ego_track_min_distance(history=history)
        statistics = {MetricStatisticsType.MAX:
                      Statistic(name="ego_max_min_distance_to_tracks", unit="meters",
                                value=np.amax(min_distances)),
                      MetricStatisticsType.MIN:
                          Statistic(name="ego_min_min_distance_to_tracks", unit="meters",
                                    value=np.amin(min_distances)),
                      MetricStatisticsType.P90:
                          Statistic(name="ego_p90_min_distance_to_tracks", unit="meters",
                                    value=np.percentile(np.abs(min_distances), 90)),  # type:ignore
                      }

        time_stamps = extract_ego_time_point(history)
        time_series = TimeSeries(unit='meters',
                                 time_stamps=list(time_stamps),
                                 values=list(min_distances))
        result = MetricStatistics(metric_computator=self.name,
                                  name="ego_min_distance_to_tracks_statistics",
                                  statistics=statistics,
                                  time_series=time_series,
                                  metric_category=self.category)

        return [result]
