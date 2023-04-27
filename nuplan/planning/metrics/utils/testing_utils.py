from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.tests.test_utils_nuplan_db import NUPLAN_MAP_VERSION, NUPLAN_MAPS_ROOT
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder
from nuplan.planning.metrics.metric_result import MetricStatistics, TimeSeries
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.utils.serialization.from_scene import from_scene_to_tracked_objects


def setup_history(scene: Dict[str, Any], scenario: MockAbstractScenario) -> SimulationHistory:
    """
    Mock the history with a mock scenario. The scenario contains the map api, and markers present in the scene are
    used to build a list of ego poses.
    :param scene: The json scene.
    :param scenario: Scenario object.
    :return The mock history.
    """
    # Update expert driving if exist in .json file
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
            scenario.get_ego_future_trajectory = lambda iteration, time_horizon, num_samples: expert_egos[
                iteration : iteration + time_horizon + 1 : time_horizon // num_samples
            ][1 : num_samples + 1]

    # Load map
    map_name = scene['map']['area']
    map_api = get_maps_api(NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, map_name)

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

    # Update the default Mock scenario duration and end_time based on the number of ego_states in the scene
    if ego_states:
        scenario.get_number_of_iterations = lambda: len(ego_states)

    # Add simulation iterations and trajectory for each iteration
    simulation_iterations = []
    trajectories = []
    for index, ego_state in enumerate(ego_states):
        simulation_iterations.append(SimulationIteration(ego_state.time_point, index))
        # Create a dummy history buffer
        history_buffer = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=10,
            ego_states=[ego_states[index]],
            observations=[observations[index]],
            sample_interval=1,
        )
        # Create trajectory using simple planner
        planner_input = PlannerInput(
            iteration=SimulationIteration(ego_states[index].time_point, 0), history=history_buffer
        )
        planner = SimplePlanner(horizon_seconds=10.0, sampling_time=1, acceleration=[0.0, 0.0])
        trajectories.append(planner.compute_planner_trajectory(planner_input))

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
    A common template to create a test history and scenario.
    :param scene: A json format to represent a scene.
    :return The mock history and scenario.
    """
    goal_pose = None
    if 'goal' in scene and 'pose' in scene['goal'] and scene['goal']['pose']:
        goal_pose = StateSE2(x=scene['goal']['pose'][0], y=scene['goal']['pose'][1], heading=scene['goal']['pose'][2])
    # Set the initial timepoint and time_step from the scene
    if (
        'ego' in scene
        and 'time_us' in scene['ego']
        and 'ego_future_states' in scene
        and scene['ego_future_states']
        and 'time_us' in scene['ego_future_states'][0]
    ):
        initial_time_us = TimePoint(time_us=scene['ego']['time_us'])
        time_step = (scene['ego_future_states'][0]['time_us'] - scene['ego']['time_us']) * 1e-6
        mock_abstract_scenario = MockAbstractScenario(initial_time_us=initial_time_us, time_step=time_step)
    else:
        mock_abstract_scenario = MockAbstractScenario()
    if goal_pose is not None:
        mock_abstract_scenario.get_mission_goal = lambda: goal_pose
    history = setup_history(scene, mock_abstract_scenario)

    return history, mock_abstract_scenario


def metric_statistic_test(
    scene: Dict[str, Any],
    metric: AbstractMetricBuilder,
    history: Optional[SimulationHistory] = None,
    mock_abstract_scenario: Optional[MockAbstractScenario] = None,
) -> MetricStatistics:
    """
    A common template to test metric statistics.
    :param scene: A json format to represent a scene.
    :param metric: An evaluation metric.
    :param history: A SimulationHistory history.
    :param mock_abstract_scenario: A scenario.
    :return Metric statistics.
    """
    if not history or not mock_abstract_scenario:
        history, mock_abstract_scenario = build_mock_history_scenario_test(scene)
    metric_results = metric.compute(history, mock_abstract_scenario)
    expected_statistics_list = scene['expected']
    if not isinstance(expected_statistics_list, list):
        expected_statistics_list = [expected_statistics_list]
    for ind, metric_result in enumerate(metric_results):
        statistics = metric_result.statistics
        expected_statistic = expected_statistics_list[ind]['statistics']
        assert len(expected_statistic) == len(statistics), (
            f"Length of actual ({len(statistics)}) and expected "
            f"({len(expected_statistic)}) statistics must be same!"
        )
        for expected_statistic, statistic in zip(expected_statistic, statistics):
            expected_type, expected_value = expected_statistic
            assert expected_type == str(statistic.type), (
                f"Statistic types don't match. Actual: {statistic.type}, " f"Expected: {expected_type}"
            )
            assert np.isclose(expected_value, statistic.value, atol=1e-2), (
                f"Statistic values don't match. Actual: {statistic.value}, " f"Expected: {expected_value}"
            )

        expected_time_series = expected_statistics_list[ind].get('time_series', None)
        if expected_time_series and metric_result.time_series is not None:
            time_series = metric_result.time_series
            expected_time_series = expected_statistics_list[ind]['time_series']
            assert isinstance(time_series, TimeSeries), 'Time series type not correct.'
            assert time_series.time_stamps == expected_time_series['time_stamps'], 'Time stamps are not correct.'
            assert np.all(
                np.round(time_series.values, 2) == expected_time_series['values']
            ), 'Time stamp values are not correct.'

    return metric_result
