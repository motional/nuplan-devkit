from __future__ import annotations

import abc
from functools import lru_cache
from typing import List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, Transform
from nuplan.planning.simulation.observation.observation_type import Detections, DetectionsTracks, Sensors


class AbstractScenario(abc.ABC):
    """
    Interface for a generic scenarios from any database.
    """

    @property
    @abc.abstractmethod
    def token(self) -> str:
        """
        Unique identifier of a scenario
        :return: str representing unique token.
        """
        pass

    @property
    @abc.abstractmethod
    def log_name(self) -> str:
        """
        Log name for from which this scenario was created
        :return: str representing log name.
        """
        pass

    @property
    @abc.abstractmethod
    def scenario_name(self) -> str:
        """
        Name of this scenario, e.g. extraction_xxxx
        :return: str representing name of this scenario.
        """
        pass

    @property
    @abc.abstractmethod
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """
        Query the vehicle parameters of ego
        :return: VehicleParameters struct.
        """
        pass

    @property
    @abc.abstractmethod
    def scenario_type(self) -> str:
        """
        :return: type of scenario e.g. [lane_change, lane_follow, ...].
        """
        pass

    @property
    @abc.abstractmethod
    def map_api(self) -> AbstractMap:
        """
        Return the Map API for this scenario
        :return: AbstractMap.
        """
        pass

    @property
    @abc.abstractmethod
    def database_interval(self) -> float:
        """
        Database interval in seconds
        :return: [s] database interval.
        """
        pass

    @abc.abstractmethod
    def get_number_of_iterations(self) -> int:
        """
        Get how many frames does this scenario contain
        :return: [int] representing number of scenarios.
        """
        pass

    @abc.abstractmethod
    def flatten(self) -> List[AbstractScenario]:
        """
        Flatten the scenario to multiple scenarios with number_of_iterations=1
        :return: List of scenarios consisting of single frame.
        """
        pass

    @abc.abstractmethod
    def get_time_point(self, iteration: int) -> TimePoint:
        """
        Get time point of the iteration
        :param iteration: iteration in scenario 0 <= iteration < number_of_iterations
        :return: global time point.
        """
        pass

    @property
    def start_time(self) -> TimePoint:
        """
        Get the start time of a scenario
        :return: starting time.
        """
        return self.get_time_point(0)

    @property
    def end_time(self) -> TimePoint:
        """
        Get end time of the scenario
        :return: end time point.
        """
        return self.get_time_point(self.get_number_of_iterations() - 1)

    @abc.abstractmethod
    def get_lidar_to_ego_transform(self) -> Transform:
        """
        Return the transformation matrix between lidar and ego
        :return: [4x4] rotation and translation matrix.
        """
        pass

    @abc.abstractmethod
    def get_mission_goal(self) -> StateSE2:
        """
        Goal far into future (in generally more than 100m far beyond scenario length).
        :return: StateSE2 for the final state.
        """
        pass

    @abc.abstractmethod
    def get_expert_goal_state(self) -> StateSE2:
        """
        Get the final state which the expert driver achieved at the end of the scenario
        :return: StateSE2 for the final state.
        """
        pass

    @abc.abstractmethod
    def get_detections_at_iteration(self, iteration: int) -> Detections:
        """
        Return detections from iteration
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: Detections.
        """
        pass

    @property
    def initial_detections(self) -> Detections:
        """
        Get initial detections
        :return: Detections.
        """
        return self.get_detections_at_iteration(0)

    @abc.abstractmethod
    def get_tracked_objects_at_iteration(self, iteration: int) -> DetectionsTracks:
        """
        Return tracked objects from iteration
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: DetectionsTracks.
        """
        pass

    @property
    def initial_tracked_objects(self) -> DetectionsTracks:
        """
        Get initial tracked objects
        :return: DetectionsTracks.
        """
        return self.get_tracked_objects_at_iteration(0)

    @abc.abstractmethod
    def get_sensors_at_iteration(self, iteration: int) -> Sensors:
        """
        Return sensor from iteration
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: Sensors.
        """
        pass

    @property
    def initial_sensors(self) -> Sensors:
        """
        Return the initial sensors (e.g. pointcloud)
        :return: Sensors.
        """
        return self.get_sensors_at_iteration(0)

    @abc.abstractmethod
    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """
        Return ego (expert) state in a dataset
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: EgoState of ego.
        """
        pass

    @property
    def initial_ego_state(self) -> EgoState:
        """
        Return the initial ego state
        :return: EgoState of ego.
        """
        return self.get_ego_state_at_iteration(0)

    @abc.abstractmethod
    def get_traffic_light_status_at_iteration(self, iteration: int) -> List[TrafficLightStatusData]:
        """
        Get traffic light status at an iteration.
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return traffic light status at the iteration.
        """
        pass

    @lru_cache(maxsize=1)
    def get_expert_ego_trajectory(self) -> List[EgoState]:
        """
        Return trajectory that was taken by the expert-driver
        :return: sequence of agent states taken by ego.
        """
        return [self.get_ego_state_at_iteration(index) for index in range(self.get_number_of_iterations())]

    def get_ego_trajectory_slice(self, start_idx: int, end_idx: int) -> List[EgoState]:
        """
        Return trajectory that was taken by the expert-driver between start_idx and end_idx
        :param start_idx: starting index for ego's trajectory
        :param end_idx: ending index for ego's trajectory
        :return: sequence of agent states taken by ego
        timestamp (best matching to the database).
        """
        return [self.get_ego_state_at_iteration(index) for index in range(start_idx, end_idx)]

    @abc.abstractmethod
    def get_future_timestamps(self, iteration: int, num_samples: int, time_horizon: float) -> List[TimePoint]:
        """
        Find timesteps in future
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the future timestamps with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_past_timestamps(self, iteration: int, num_samples: int, time_horizon: float) -> List[TimePoint]:
        """
        Find timesteps in past
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the past
        :param time_horizon: the desired horizon to the past
        :return: the future timestamps with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_ego_future_trajectory(self, iteration: int, num_samples: int, time_horizon: float) -> List[
            EgoState]:
        """
        Find ego future trajectory
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the future ego trajectory with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_ego_past_trajectory(self, iteration: int, num_samples: int, time_horizon: float) -> List[EgoState]:
        """
        Find ego past trajectory
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the past ego trajectory with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_past_detections(self, iteration: int, num_samples: int, time_horizon: float) -> List[Detections]:
        """
        Find past detections
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the past detections with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_future_detections(self, iteration: int, num_samples: int, time_horizon: float) -> List[Detections]:
        """
        Find future detections
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the future detections with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass

    @abc.abstractmethod
    def get_past_sensors(self, iteration: int, num_samples: int, time_horizon: float) -> List[Sensors]:
        """
        Find past sensors
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the past sensors with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        pass
