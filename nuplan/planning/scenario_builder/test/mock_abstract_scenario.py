from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.test.test_utils import get_sample_agent
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.common.maps.abstract_map_objects import AbstractMapObject
from nuplan.common.maps.maps_datatypes import LaneSegmentConnections, LaneSegmentCoords, LaneSegmentMetaData, \
    RasterLayer, RasterMap, TrafficLightStatusData, Transform
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import Detections, DetectionsTracks, Sensors
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration


def get_num_samples(num_samples: Optional[int], time_horizon: float, database_interval: float) -> int:
    """
    Set num samples based on the time_horizon and database interval if num_samples is not set
    :param num_samples: if None, it will be computed based on  math.floor(time_horizon / database_interval)
    :param time_horizon: [s] horizon in which we want to look into
    :param database_interval: interval of the database
    :return: number of samples to iterate over
    """
    return num_samples if num_samples else int(time_horizon / database_interval)


class MockMapFactory(AbstractMapFactory):

    def build_map_from_name(self, map_name: str) -> AbstractMap:
        """ Implemented. See interface. """
        return MockAbstractMap()


class MockAbstractMap(AbstractMap):
    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        """ Implemented. See interface. """
        raise NotImplementedError

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        """ Implemented. See interface. """
        return RasterMap({})

    @property
    def map_name(self) -> str:
        """ Implemented. See interface. """
        return "map_name"

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """ Implemented. See interface. """
        raise NotImplementedError

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        """ Implemented. See interface. """
        raise NotImplementedError

    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[AbstractMapObject]:
        """ Implemented. See interface. """
        raise NotImplementedError

    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[AbstractMapObject]:
        """ Implemented. See interface. """
        raise NotImplementedError

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        """ Implemented. See interface. """
        raise NotImplementedError

    def get_proximal_map_objects(self, point: Point2D, radius: float, layers: List[SemanticMapLayer]) -> Dict[
            SemanticMapLayer, List[AbstractMapObject]]:
        """ Implemented. See interface. """
        return defaultdict(list)

    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> AbstractMapObject:
        """ Implemented. See interface. """
        raise NotImplementedError

    def get_distance_to_nearest_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Tuple[int, float]:
        """ Implemented. See interface. """
        raise NotImplementedError

    def get_neighbor_vector_map(self, point: Point2D, radius: float) -> \
            Tuple[LaneSegmentCoords, LaneSegmentConnections, LaneSegmentMetaData]:
        """ Implemented. See interface. """
        raise NotImplementedError


class MockAbstractScenario(AbstractScenario):

    def __init__(self,
                 initial_time_us: TimePoint = TimePoint(time_us=1621641671099),
                 time_step: float = 0.5,
                 number_of_future_iterations: int = 10,
                 number_of_past_iterations: int = 0,
                 fixed_velocity: StateVector2D = StateVector2D(x=1.0, y=0.0),
                 number_of_detections: int = 10,
                 initial_ego_state: StateSE2 = StateSE2(x=0., y=0., heading=0.0),
                 mission_goal: StateSE2 = StateSE2(10, 0, 0)):
        """
        Create mocked scenario where ego just goes straight with fixed velocity [m/s]
        :param initial_time_us: initial time from start point of scenario [us]
        :param time_step: time step in [s]
        :param number_of_future_iterations: number of iterations in the future
        :param number_of_past_iterations: number of iterations in the past
        :param fixed_velocity: [m/s] fixed velocity
        :param number_of_detections: number of detections in the scenario
        :param initial_ego_state: Initial state of ego
        :param mission_goal: Dummy mission goal

        """
        self._initial_time_us = initial_time_us
        self._time_step = time_step
        self._number_of_past_iterations = number_of_past_iterations
        self._number_of_future_iterations = number_of_future_iterations
        self._current_iteration = number_of_past_iterations
        self._total_iterations = number_of_past_iterations + number_of_future_iterations + 1

        # Create dummy ego trajectory
        acceleration = StateVector2D(0., 0.)
        start_time_us = max(TimePoint(int(number_of_past_iterations * time_step * 1e6)), initial_time_us)
        time_horizon = self._total_iterations * time_step
        history_buffer = SimulationHistoryBuffer(buffer_size=10)
        history_buffer.append(EgoState.from_raw_params(StateSE2(x=initial_ego_state.x, y=initial_ego_state.y,
                                                                heading=initial_ego_state.heading),
                                                       time_point=start_time_us,
                                                       velocity_2d=fixed_velocity, tire_steering_angle=0.0,
                                                       acceleration_2d=acceleration),
                              Detections([]))
        self._ego_states = SimplePlanner(horizon_seconds=time_horizon,
                                         sampling_time=time_step,
                                         acceleration=acceleration.array).compute_trajectory(
            iteration=SimulationIteration(start_time_us, 0), history=history_buffer).get_sampled_trajectory()

        def _create_box(token: str) -> Box3D:
            """
            :param token: a unique instance token
            :return: a random Box3D
            """
            box = Box3D.make_random()
            box.token = token
            box.label = 1
            return box

        self._detections = [Detections(boxes=[_create_box(idx) for idx in range(number_of_detections)])  # type: ignore
                            for _ in range(self._total_iterations)]
        self._tracked_objects = [
            DetectionsTracks(TrackedObjects(agents=[get_sample_agent(str(idx)) for idx in range(number_of_detections)]))
            for _ in range(self._total_iterations)]
        self._sensors = [Sensors(pointcloud=[np.eye(3) for _ in range(number_of_detections)])
                         for _ in range(self._total_iterations)]

        if len(self._ego_states) != len(self._detections) or len(self._ego_states) != self._total_iterations:
            raise RuntimeError("The dimensions of detections and ego trajectory is not the same!")

        # Create dummy mission goal
        self._mission_goal = mission_goal

        # Create mocked map api
        self._map_api = MockAbstractMap()

    @property
    def token(self) -> str:
        """ Implemented. See interface. """
        return "mock_token"

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """ Inherited, see superclass. """
        return get_pacifica_parameters()

    @property
    def log_name(self) -> str:
        """ Implemented. See interface. """
        return "mock_log_name"

    @property
    def scenario_name(self) -> str:
        """ Implemented. See interface. """
        return "mock_scenario_name"

    @property
    def scenario_type(self) -> str:
        """ Implemented. See interface. """
        return "mock_scenario_type"

    @property
    def map_api(self) -> AbstractMap:
        """ Implemented. See interface. """
        return self._map_api

    @property
    def database_interval(self) -> float:
        """ Inherited, see superclass. """
        return self._time_step

    def get_number_of_iterations(self) -> int:
        """ Implemented. See interface. """
        return self._number_of_future_iterations

    def flatten(self) -> List[AbstractScenario]:
        """ Implemented. See interface. """
        raise NotImplementedError

    def get_time_point(self, iteration: int) -> TimePoint:
        """ Implemented. See interface. """
        return self._ego_states[self._current_iteration + iteration].time_point

    def get_lidar_to_ego_transform(self) -> Transform:
        """ Implemented. See interface. """
        return np.eye(4)

    def get_mission_goal(self) -> Optional[StateSE2]:
        """ Implemented. See interface. """
        return self._mission_goal

    def get_expert_goal_state(self) -> StateSE2:
        """ Implemented. See interface. """
        return self._mission_goal

    def get_detections_at_iteration(self, iteration: int) -> Detections:
        """ Implemented. See interface. """
        return self._detections[self._current_iteration + iteration]

    def get_tracked_objects_at_iteration(self, iteration: int) -> DetectionsTracks:
        """ Implemented. See interface. """
        return self._tracked_objects[self._current_iteration + iteration]

    def get_sensors_at_iteration(self, iteration: int) -> Sensors:
        """ Implemented. See interface. """
        raise NotImplementedError

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """ Implemented. See interface. """
        return self._ego_states[self._current_iteration + iteration]

    def get_future_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> List[
            TimePoint]:
        """ Implemented. See interface. """
        ego_states = self.get_ego_future_trajectory(iteration=iteration, time_horizon=time_horizon,
                                                    num_samples=num_samples)
        time_points = [state.time_point for state in ego_states]
        return time_points

    def get_ego_future_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> \
            List[EgoState]:
        """ Implemented. See interface. """
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._number_of_future_iterations - iteration >= indices[-1], \
            f"Requested time horizon of {time_horizon}s is too long! " \
            f"Scenario future has length {(self._number_of_future_iterations - iteration) * self._time_step}s from " \
            f"the iteration {iteration}"
        ego_states = [self._ego_states[self._current_iteration + iteration + idx] for idx in indices]
        return ego_states

    def get_past_timestamps(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> \
            List[TimePoint]:
        """ Implemented. See interface. """
        ego_states = self.get_ego_past_trajectory(iteration=iteration, time_horizon=time_horizon,
                                                  num_samples=num_samples)
        time_points = [state.time_point for state in ego_states]
        return time_points

    def get_ego_past_trajectory(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> \
            List[EgoState]:
        """ Implemented. See interface. """
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._current_iteration + iteration >= indices[-1], \
            f"Requested time horizon of {time_horizon}s is too long! " \
            f"Scenario past has length {(self._current_iteration + iteration) * self._time_step}s from " \
            f"the iteration {iteration}"
        ego_states = [self._ego_states[self._current_iteration + iteration - idx] for idx in reversed(indices)]
        return ego_states

    def get_past_detections(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> List[
            Detections]:
        """ Implemented. See interface. """
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._current_iteration + iteration >= indices[-1], \
            f"Requested time horizon of {time_horizon}s is too long! " \
            f"Scenario past has length {(self._current_iteration + iteration) * self._time_step}s from " \
            f"the iteration {iteration}"
        detections = [self._detections[self._current_iteration + iteration - idx] for idx in reversed(indices)]
        return detections

    def get_past_tracked_objects(self, iteration: int, num_samples: int, time_horizon: float) -> List[DetectionsTracks]:
        """ Implemented. See interface. """
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._current_iteration + iteration >= indices[-1], \
            f"Requested time horizon of {time_horizon}s is too long! " \
            f"Scenario past has length {(self._current_iteration + iteration) * self._time_step}s from " \
            f"the iteration {iteration}"
        detections = [self._tracked_objects[self._current_iteration + iteration - idx] for idx in reversed(indices)]
        return detections

    def get_past_sensors(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> List[Sensors]:
        """ Implemented. See interface. """
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        _sensors = [self._sensors[self._current_iteration + iteration - idx - 1] for idx in indices]
        return _sensors

    def get_future_detections(self, iteration: int, time_horizon: float, num_samples: Optional[int] = None) -> List[
            Detections]:
        """ Implemented. See interface. """
        num_samples = get_num_samples(num_samples, time_horizon, self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._number_of_future_iterations - iteration >= indices[-1], \
            f"Requested time horizon of {time_horizon}s is too long! " \
            f"Scenario future has length {(self._number_of_future_iterations - iteration) * self._time_step}s from " \
            f"the iteration {iteration}"
        detections = [self._detections[self._current_iteration + iteration + idx] for idx in indices]
        return detections

    def get_future_tracked_objects(self, iteration: int, num_samples: int, time_horizon: float) -> List[
            DetectionsTracks]:
        """ Implemented. See interface. """
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, self._time_step)
        assert self._number_of_future_iterations - iteration >= indices[-1], \
            f"Requested time horizon of {time_horizon}s is too long! " \
            f"Scenario future has length {(self._number_of_future_iterations - iteration) * self._time_step}s from " \
            f"the iteration {iteration}"
        detections = [self._tracked_objects[self._current_iteration + iteration + idx] for idx in indices]
        return detections

    def get_traffic_light_status_at_iteration(self, iteration: int) -> List[TrafficLightStatusData]:
        """ Implemented. see interface. """

        return []
