from typing import List, Type

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import Detections, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory


def _f_dot_kinematic_car(state: EgoState, vehicle: VehicleParameters) -> EgoState:
    """
    Compute x_dot = f(x) for a kinematic car

    :param state for which to compute motion model
    :param vehicle parameters
    """
    lf = vehicle.front_length - vehicle.cog_position_from_rear_axle
    lr = vehicle.rear_length + vehicle.cog_position_from_rear_axle
    beta = np.arctan2(lr * np.tan(state.tire_steering_angle), (lr + lf))

    car_speed = np.hypot(state.dynamic_car_state.rear_axle_velocity_2d.x,
                         state.dynamic_car_state.rear_axle_velocity_2d.y)

    return EgoState.from_raw_params(
        pose=StateSE2(x=car_speed * np.cos(beta + state.rear_axle.heading),
                      y=car_speed * np.sin(beta + state.rear_axle.heading),
                      heading=(car_speed / lr) * np.sin(beta)),
        velocity_2d=state.dynamic_car_state.rear_axle_acceleration_2d,
        acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=state.time_point)


class SimplePlanner(AbstractPlanner):
    """
    Planner going straight
    """

    def __init__(self,
                 horizon_seconds: float,
                 sampling_time: float,
                 acceleration: npt.NDArray[np.float32],
                 max_velocity: float = 5.0,
                 steering_angle: float = 0.0):
        self.horizon_seconds = horizon_seconds
        self.sampling_time = sampling_time
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle

    def initialize(self,
                   expert_goal_state: StateSE2,
                   mission_goal: StateSE2,
                   map_name: str,
                   map_api: AbstractMap) -> None:
        """ Inherited, see superclass. """
        pass

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """ Inherited, see superclass. """
        return Detections  # type: ignore

    def compute_trajectory(self, iteration: SimulationIteration,
                           history: SimulationHistoryBuffer) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """
        ego_state = history.ego_states[-1]
        state = EgoState(car_footprint=ego_state.car_footprint,
                         dynamic_car_state=DynamicCarState(ego_state.car_footprint.rear_axle_to_center_dist,
                                                           ego_state.dynamic_car_state.rear_axle_velocity_2d,
                                                           self.acceleration),
                         tire_steering_angle=self.steering_angle,
                         time_point=ego_state.time_point)
        trajectory: List[EgoState] = [state]
        for time_in_future in np.arange(iteration.time_us + self.sampling_time * 1e6,
                                        iteration.time_us + self.horizon_seconds * 1e6,
                                        self.sampling_time * 1e6):

            state_dot = _f_dot_kinematic_car(state, get_pacifica_parameters())
            next_point_x = state.rear_axle.x + state_dot.rear_axle.x * self.sampling_time
            next_point_y = state.rear_axle.y + state_dot.rear_axle.y * self.sampling_time
            next_point_heading = state.rear_axle.heading + state_dot.rear_axle.heading * self.sampling_time

            if state.dynamic_car_state.speed > self.max_velocity:
                state_dot.velocity_x = 0.0
                state_dot.velocity_y = 0.0

            next_point_velocity_x = np.fmax(0, state.dynamic_car_state.rear_axle_velocity_2d.x +
                                            state_dot.dynamic_car_state.rear_axle_acceleration_2d.x *
                                            self.sampling_time)
            next_point_velocity_y = np.fmax(0,
                                            state.dynamic_car_state.rear_axle_velocity_2d.y +
                                            state_dot.dynamic_car_state.rear_axle_acceleration_2d.y *
                                            self.sampling_time)
            state = EgoState.from_raw_params(
                StateSE2(next_point_x, next_point_y, next_point_heading),
                velocity_2d=StateVector2D(next_point_velocity_x,
                                          next_point_velocity_y),
                acceleration_2d=state.dynamic_car_state.rear_axle_acceleration_2d,
                tire_steering_angle=state.tire_steering_angle,
                time_point=TimePoint(time_in_future))

            trajectory.append(state)

        return InterpolatedTrajectory(trajectory)
