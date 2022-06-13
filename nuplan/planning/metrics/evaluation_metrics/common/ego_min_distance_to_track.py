from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, Statistic, TimeSeries
from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


class EgoMinDistanceToTrackStatistics(MetricBase):
    """Ego minimum distance to any tracks metric."""

    def __init__(self, name: str, category: str) -> None:
        """
        Initializes the EgoMinDistanceToTrackStatistics class
        :param name: Metric name
        :param category: Metric category.
        """
        super().__init__(name=name, category=category)

    @staticmethod
    def extract_tracks(history_samples: List[SimulationHistorySample]) -> List[Optional[List[Polygon]]]:
        """
        Extract tracks based on detections
        :param history_samples: A list of history samples in scenario
        :return A list of arrays of box/track centers.
        """
        tracks: List[Optional[List[Polygon]]] = []
        for history_sample in history_samples:
            detections = history_sample.observation
            if not isinstance(detections, DetectionsTracks):
                tracks.append(None)
            else:
                tracked_objects = detections.tracked_objects
                track_polygons = [tracked_objects.box.geometry for tracked_objects in tracked_objects]
                tracks.append(track_polygons)

        return tracks

    def compute_ego_track_min_distance(self, history: SimulationHistory) -> npt.NDArray[np.float32]:
        """
        Compute minimum distance of ego poses and tracks (detections) in a scenario
        :param history: History from a simulation engine
        :return An array of absolute minimum distances between ego poses and tracks in a scenario.
        """
        # Extract trajectory of each box.
        scenario_tracks = self.extract_tracks(history.data)

        # Extract ego pose center x and y.
        ego_pose_polygons = [sample.ego_state.car_footprint.oriented_box.geometry for sample in history.data]

        min_distances = []
        assert len(scenario_tracks) == len(ego_pose_polygons), 'Length of tracks and ego poses should be the same.'

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

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: Dict[str, Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        # TODO: Define the metric score
        return 0.0

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the minimum distance to all tracks metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated minimum distance to all tracks metric.
        """
        min_distances = self.compute_ego_track_min_distance(history=history)
        time_stamps = extract_ego_time_point(history.extract_ego_state)
        time_series = TimeSeries(unit='meters', time_stamps=list(time_stamps), values=list(min_distances))
        metric_statistics = self._compute_time_series_statistic(time_series=time_series)
        results = self._construct_metric_results(
            metric_statistics=metric_statistics, time_series=time_series, scenario=scenario
        )
        return results  # type: ignore
