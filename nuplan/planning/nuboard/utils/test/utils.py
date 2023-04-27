from pathlib import Path

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


def create_sample_simulation_log(output_path: Path) -> SimulationLog:
    """
    Generates a sample simulation log for use in tests.
    :param output_path: to write to.
    """
    scenario = MockAbstractScenario()

    planner = SimplePlanner(horizon_seconds=2, sampling_time=0.5, acceleration=np.array([0.0, 0.0]))

    # Mock two iteration steps
    history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
    state_0 = EgoState.build_from_rear_axle(
        StateSE2(0, 0, 0),
        vehicle_parameters=scenario.ego_vehicle_parameters,
        rear_axle_velocity_2d=StateVector2D(x=0, y=0),
        rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
        tire_steering_angle=0,
        time_point=TimePoint(0),
    )
    state_1 = EgoState.build_from_rear_axle(
        StateSE2(0, 0, 0),
        vehicle_parameters=scenario.ego_vehicle_parameters,
        rear_axle_velocity_2d=StateVector2D(x=0, y=0),
        rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
        tire_steering_angle=0,
        time_point=TimePoint(1000),
    )
    history.add_sample(
        SimulationHistorySample(
            iteration=SimulationIteration(time_point=TimePoint(0), index=0),
            ego_state=state_0,
            trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
            observation=DetectionsTracks(TrackedObjects()),
            traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0)),
        )
    )
    history.add_sample(
        SimulationHistorySample(
            iteration=SimulationIteration(time_point=TimePoint(0), index=0),
            ego_state=state_1,
            trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
            observation=DetectionsTracks(TrackedObjects()),
            traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0)),
        )
    )

    return SimulationLog(file_path=Path(output_path), scenario=scenario, planner=planner, simulation_history=history)
