import os
from typing import Any, Dict, List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from nuplan.planning.utils.serialization.from_scene import from_scene_to_agents

NUPLAN_DB_VERSION = 'nuplan-maps-v0.1'


def _get_simple_agent_state(pose: List[float], time_point: TimePoint) -> EgoState:
    """
    Builds a mock agent state containing only the time point and the pose

    :param pose: The current pose
    :param time_point: The time point relative to the pose
    :return An agent state
    """
    return EgoState.from_raw_params(time_point=time_point, pose=StateSE2(x=pose[0], y=pose[1], heading=pose[2]),
                                    velocity_2d=StateVector2D(0.0, 0.0),
                                    acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=0.0)


def setup_history(scene: Dict[str, Any]) -> SimulationHistory:
    """
    Mocks the history with a mock scenario. The scenario contains the map api, and markers present in the scene are
    used to build a list of ego poses.

    :param scene: The json scene.
    :return The mock history.
    """

    maps_db = GPKGMapsDB(NUPLAN_DB_VERSION,
                         map_root=os.path.join(os.getenv('NUPLAN_DATA_ROOT', "~/nuplan/dataset"), 'maps'))

    # Load map
    map_name = scene["map"]["area"]
    map_api = NuPlanMapFactory(maps_db).build_map_from_name(map_name)

    # Extract Agent Box
    boxes = from_scene_to_agents(scene["world"])
    for id, box in enumerate(boxes):
        box.token = str(id + 1)
    ego_pose = scene["ego"]["pose"]
    ego_x = ego_pose[0]
    ego_y = ego_pose[1]
    ego_heading = ego_pose[2]

    ego_state = EgoState.from_raw_params(time_point=TimePoint(0),
                                         pose=StateSE2(x=ego_x,
                                                       y=ego_y,
                                                       heading=ego_heading),
                                         velocity_2d=StateVector2D(x=scene["ego"]["speed"], y=0.0),
                                         acceleration_2d=StateVector2D(x=scene["ego"]["acceleration"], y=0.0),
                                         tire_steering_angle=0)

    scenario = MockAbstractScenario()
    history = SimulationHistory(map_api, scenario.get_mission_goal())
    history.add_sample(SimulationHistorySample(
        iteration=SimulationIteration(TimePoint(0), 0),
        ego_state=ego_state,
        trajectory=InterpolatedTrajectory([ego_state, ego_state]),
        observation=Detections(boxes=boxes)
    ))

    ego_future_states: List[Dict[str, Any]] = scene["ego_future_states"] \
        if "ego_future_states" in scene else []

    for index, ego_future_state in enumerate(ego_future_states):
        pose = ego_future_state["pose"]
        ego_state = EgoState.from_raw_params(time_point=TimePoint(index + 1),
                                             pose=StateSE2(x=pose[0],
                                                           y=pose[1],
                                                           heading=pose[2]),
                                             velocity_2d=StateVector2D(x=ego_future_state["speed"], y=0.0),
                                             acceleration_2d=StateVector2D(x=ego_future_state["acceleration"],
                                                                           y=0.0),
                                             tire_steering_angle=0)
        history.add_sample(SimulationHistorySample(
            iteration=SimulationIteration(TimePoint(index + 1), index + 1),
            ego_state=ego_state,
            trajectory=InterpolatedTrajectory([ego_state, ego_state]),
            observation=Detections(boxes=boxes)
        ))

    return history
