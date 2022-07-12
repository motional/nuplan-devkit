from typing import Any, Dict, List, Tuple

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.database.tests.nuplan_db_test_utils import get_test_maps_db
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, TimeSeries
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.utils.serialization.from_scene import from_scene_to_tracked_objects


def setup_history(scene: Dict[str, Any], scenario: MockAbstractScenario) -> SimulationHistory:
    """
    Mocks the history with a mock scenario. The scenario contains the map api, and markers present in the scene are
    used to build a list of ego poses
    :param scene: The json scene
    :param scenario: Scenario object
    :return The mock history.
    """
    # Update expert driving if there is
    if 'expert_ego_states' in scene:
        expert_ego_states = scene['expert_ego_states']
        expert_egos = []

        for expert_ego_state in expert_ego_states:
            ego_state = EgoState.build_from_rear_axle(
                time_point=TimePoint(expert_ego_state['time_us']),
                rear_axle_pose=StateSE2(
                    x=expert_ego_state['pose'][0], y=expert_ego_state['pose'][1], heading=expert_ego_state['pose'][2]
                ),
                rear_axle_velocity_2d=StateVector2D(
                    x=expert_ego_state['velocity'][0], y=expert_ego_state['velocity'][1]
                ),
                rear_axle_acceleration_2d=StateVector2D(
                    x=expert_ego_state['acceleration'][0], y=expert_ego_state['acceleration'][1]
                ),
                tire_steering_angle=0,
                vehicle_parameters=scenario.ego_vehicle_parameters,
            )
            expert_egos.append(ego_state)

        if len(expert_egos):
            scenario.get_expert_ego_trajectory = lambda: expert_egos

    maps_db = get_test_maps_db()

    # Load map
    map_name = scene['map']['area']
    map_api = NuPlanMapFactory(maps_db).build_map_from_name(map_name)

    # Extract Agent Box
    tracked_objects = from_scene_to_tracked_objects(scene['world'])
    for tracked_object in tracked_objects:
        tracked_object._track_token = tracked_object.token

    ego_pose = scene['ego']['pose']
    ego_x = ego_pose[0]
    ego_y = ego_pose[1]
    ego_heading = ego_pose[2]

    # Add both ego states and agents in the current timestamps
    ego_states = []
    observations = []
    ego_state = EgoState.build_from_rear_axle(
        time_point=TimePoint(scene['ego']['time_us']),
        rear_axle_pose=StateSE2(x=ego_x, y=ego_y, heading=ego_heading),
        rear_axle_velocity_2d=StateVector2D(x=scene['ego']['velocity'][0], y=scene['ego']['velocity'][1]),
        rear_axle_acceleration_2d=StateVector2D(x=scene['ego']['acceleration'][0], y=scene['ego']['acceleration'][1]),
        tire_steering_angle=0,
        vehicle_parameters=scenario.ego_vehicle_parameters,
    )
    ego_states.append(ego_state)
    observations.append(DetectionsTracks(tracked_objects))

    # Add both ego states and agents in the future timestamp
    ego_future_states: List[Dict[str, Any]] = scene['ego_future_states'] if 'ego_future_states' in scene else []
    world_future_states: List[Dict[str, Any]] = scene['world_future_states'] if 'world_future_states' in scene else []
    assert len(ego_future_states) == len(world_future_states), (
        f'Length of world world_future_states: '
        f'{len(world_future_states)} and '
        f'length of ego_future_states: '
        f'{len(ego_future_states)} not same'
    )
    for index, (ego_future_state, future_world_state) in enumerate(zip(ego_future_states, world_future_states)):
        pose = ego_future_state['pose']
        time_us = ego_future_state['time_us']
        ego_state = EgoState.build_from_rear_axle(
            time_point=TimePoint(time_us),
            rear_axle_pose=StateSE2(x=pose[0], y=pose[1], heading=pose[2]),
            rear_axle_velocity_2d=StateVector2D(x=ego_future_state['velocity'][0], y=ego_future_state['velocity'][1]),
            rear_axle_acceleration_2d=StateVector2D(
                x=ego_future_state['acceleration'][0], y=ego_future_state['acceleration'][1]
            ),
            vehicle_parameters=scenario.ego_vehicle_parameters,
            tire_steering_angle=0,
        )
        future_tracked_objects = from_scene_to_tracked_objects(future_world_state)
        for future_tracked_object in future_tracked_objects:
            future_tracked_object._track_token = future_tracked_object.token

        ego_states.append(ego_state)
        observations.append(DetectionsTracks(future_tracked_objects))

    # Add simulation iterations and trajectories based on a series of ego states
    simulation_iterations = []
    trajectories = []
    for index, ego_state in enumerate(ego_states):
        simulation_iterations.append(SimulationIteration(ego_state.time_point, index))
        if (index + 1) < len(ego_states):
            next_ego_state = ego_states[index + 1]
        else:
            repeated_last_state = EgoState.build_from_rear_axle(
                time_point=TimePoint(ego_states[-1].time_us + 1000000),
                rear_axle_pose=StateSE2(
                    x=ego_states[-1].rear_axle.x, y=ego_states[-1].rear_axle.y, heading=ego_states[-1].rear_axle.heading
                ),
                rear_axle_velocity_2d=StateVector2D(
                    x=ego_states[-1].dynamic_car_state.rear_axle_velocity_2d.x,
                    y=ego_states[-1].dynamic_car_state.rear_axle_velocity_2d.y,
                ),
                rear_axle_acceleration_2d=StateVector2D(
                    x=ego_states[-1].dynamic_car_state.rear_axle_acceleration_2d.x,
                    y=ego_states[-1].dynamic_car_state.rear_axle_acceleration_2d.y,
                ),
                vehicle_parameters=scenario.ego_vehicle_parameters,
                tire_steering_angle=ego_states[-1].tire_steering_angle,
            )
            next_ego_state = repeated_last_state

        trajectories.append(InterpolatedTrajectory([ego_state, next_ego_state]))

    # Create simulation histories
    history = SimulationHistory(map_api, scenario.get_mission_goal())
    for ego_state, simulation_iteration, trajectory, observation in zip(
        ego_states, simulation_iterations, trajectories, observations
    ):
        history.add_sample(
            SimulationHistorySample(
                iteration=simulation_iteration,
                ego_state=ego_state,
                trajectory=trajectory,
                observation=observation,
                traffic_light_status=scenario.get_traffic_light_status_at_iteration(simulation_iteration.index),
            )
        )

    return history


def build_mock_history_scenario_test(scene: Dict[str, Any]) -> Tuple[SimulationHistory, MockAbstractScenario]:
    """
    A common template to create a test history and scenario
    :param scene: A json format to represent a scene
    :return The mock history and scenario
    """
    goal_pose = None
    if 'goal' in scene and 'pose' in scene['goal'] and scene['goal']['pose']:
        goal_pose = StateSE2(x=scene['goal']['pose'][0], y=scene['goal']['pose'][1], heading=scene['goal']['pose'][2])
    mock_abstract_scenario = MockAbstractScenario()
    if goal_pose is not None:
        mock_abstract_scenario.get_mission_goal = lambda: goal_pose
    history = setup_history(scene, scenario=mock_abstract_scenario)

    return history, mock_abstract_scenario


def metric_statistic_test(scene: Dict[str, Any], metric: AbstractMetricBuilder) -> MetricStatistics:
    """
    A common template to test metric statistics
    :param scene: A json format to represent a scene
    :param metric: An evaluation metric
    :return Metric statistics.
    """
    history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    metric_result = metric.compute(history, mock_abstract_scenario)[0]
    statistics = metric_result.statistics
    expected_statistics = scene['expected']['statistics']
    for statistics_type, expected_value in expected_statistics.items():
        value = statistics[MetricStatisticsType.__members__[statistics_type.upper()]].value
        if statistics_type.upper() not in [str(MetricStatisticsType.BOOLEAN)]:
            value = np.round(value, 2)
        assert value == expected_value, f'Statistic value incorrect: Actual: {value},  Expected: {expected_value}.'

    expected_time_series = scene['expected'].get('time_series', None)
    if expected_time_series and metric_result.time_series is not None:
        time_series = metric_result.time_series
        expected_time_series = scene['expected']['time_series']
        assert isinstance(time_series, TimeSeries), 'Time series type not correct.'
        assert time_series.time_stamps == expected_time_series['time_stamps'], 'Time stamps are not correct.'
        assert np.all(
            np.round(time_series.values, 2) == expected_time_series['values']
        ), 'Time stamp values are not correct.'

    return metric_result
