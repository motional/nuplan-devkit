from typing import List, Type

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class SimplePlanner(AbstractPlanner):
    """
    Planner going straight.
    """

    def __init__(
        self,
        horizon_seconds: float,
        sampling_time: float,
        acceleration: npt.NDArray[np.float32],
        max_velocity: float = 5.0,
        steering_angle: float = 0.0,
    ):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param acceleration: [m/s^2] constant ego acceleration, till limited by max_velocity.
        :param max_velocity: [m/s] ego max velocity.
        :param steering_angle: [rad] ego steering angle.
        """
        self.horizon_seconds = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""
        pass

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """
        # Extract current state
        history = current_input.history
        ego_state, _ = history.current_state
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                self.acceleration,
            ),
            tire_steering_angle=self.steering_angle,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        trajectory: List[EgoState] = [state]
        for _ in range(int(self.horizon_seconds.time_us / self.sampling_time.time_us)):
            if state.dynamic_car_state.speed > self.max_velocity:
                accel = self.max_velocity - state.dynamic_car_state.speed
                state = EgoState.build_from_rear_axle(
                    rear_axle_pose=state.rear_axle,
                    rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
                    rear_axle_acceleration_2d=StateVector2D(accel, 0),
                    tire_steering_angle=state.tire_steering_angle,
                    time_point=state.time_point,
                    vehicle_parameters=state.car_footprint.vehicle_parameters,
                    is_in_auto_mode=True,
                    angular_vel=state.dynamic_car_state.angular_velocity,
                    angular_accel=state.dynamic_car_state.angular_acceleration,
                )

            state = self.motion_model.propagate_state(state, state.dynamic_car_state, self.sampling_time)
            trajectory.append(state)

        return InterpolatedTrajectory(trajectory)
