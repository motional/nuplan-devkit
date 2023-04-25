from typing import Any, Generator, List, Optional, Set, Tuple, Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatuses, Transform
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, SensorChannel, Sensors
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class CachedScenario(AbstractScenario):
    """
    A class representing a cached scenario.
    This class is backend-agnostic, and serves as a pointer to precomputed features.
    """

    def __init__(self, log_name: str, token: str, scenario_type: str) -> None:
        """
        Construct a cached scenario objet.
        :param log_name: The log name for the scenario.
        :param token: The token for the scenario.
        :param scenario_type: The scenario type.
        """
        self._log_name = log_name
        self._token = token
        self._scenario_type = scenario_type

    def __reduce__(self) -> Tuple[Type['CachedScenario'], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (
            self.__class__,
            (self._log_name, self._token, self._scenario_type),
        )

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._token

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement scenario_name.")

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement ego_vehicle_parameters.")

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_type

    @property
    def map_api(self) -> str:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement map_api.")

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement database_interval.")

    @property
    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_number_of_iterations.")

    def get_time_point(self) -> TimePoint:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_time_point.")

    def get_lidar_to_ego_transform(self) -> Transform:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_lidar_to_ego_transform.")

    def get_mission_goal(self) -> StateSE2:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_mission_goal.")

    def get_route_roadblock_ids(self) -> List[str]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_route_roadblock_ids.")

    def get_expert_goal_state(self) -> StateSE2:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_expert_goal_state.")

    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_tracked_objects_at_iteration.")

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        raise NotImplementedError(
            "CachedScenario does not implement get_tracked_objects_within_time_window_at_iteration."
        )

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]] = None) -> Sensors:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_sensors_at_iteration.")

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_ego_state_at_iteration.")

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_traffic_light_status_at_iteration.")

    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_past_traffic_light_status_history.")

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_future_traffic_light_status_history.")

    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_future_timestamps.")

    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_past_timestamps.")

    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_ego_future_trajectory.")

    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_ego_past_trajectory.")

    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_past_sensors.")

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_past_tracked_objects.")

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError("CachedScenario does not implement get_future_tracked_objects.")
