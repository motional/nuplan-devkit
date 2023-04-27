import asyncio
import itertools
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nest_asyncio
import numpy as np
import numpy.typing as npt
from bokeh.document.document import Document
from bokeh.io import show
from bokeh.io.state import curstate
from bokeh.layouts import column

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_lidarpc_tokens_with_scenario_tag_from_db,
    get_sensor_data_token_timestamp_from_db,
    get_sensor_token_map_name_from_db,
)
from nuplan.planning.nuboard.base.data_class import NuBoardFile, SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.base.simulation_tile import SimulationTile
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

logger = logging.getLogger(__name__)


@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def construct_nuboard_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to nuBoard configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, 'config', 'common')
    config_name = 'default_nuboard'
    config_path = join(base_config_path, 'config/nuboard')
    experiment_dir = "file://" + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


def construct_simulation_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = join(base_config_path, 'config', 'simulation')
    experiment_dir = "file://" + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


def save_scenes_to_dir(
    scenario: AbstractScenario, save_dir: str, simulation_history: SimulationHistory
) -> SimulationScenarioKey:
    """
    Save scenes to a directory.
    :param scenario: Scenario.
    :param save_dir: Save path.
    :param simulation_history: Simulation history.
    :return: Scenario key of simulation.
    """
    planner_name = "tutorial_planner"
    scenario_type = scenario.scenario_type
    scenario_name = scenario.scenario_name
    log_name = scenario.log_name

    save_path = Path(save_dir)
    file = save_path / planner_name / scenario_type / log_name / scenario_name / (scenario_name + ".msgpack.xz")
    file.parent.mkdir(exist_ok=True, parents=True)

    # Create a dummy planner
    dummy_planner = _create_dummy_simple_planner(acceleration=[5.0, 5.0])
    simulation_log = SimulationLog(
        planner=dummy_planner, scenario=scenario, simulation_history=simulation_history, file_path=file
    )
    simulation_log.save_to_file()

    return SimulationScenarioKey(
        planner_name=planner_name,
        scenario_name=scenario_name,
        scenario_type=scenario_type,
        nuboard_file_index=0,
        log_name=log_name,
        files=[file],
    )


def _create_dummy_simple_planner(
    acceleration: List[float], horizon_seconds: float = 10.0, sampling_time: float = 20.0
) -> SimplePlanner:
    """
    Create a dummy simple planner.
    :param acceleration: [m/s^2] constant ego acceleration, till limited by max_velocity.
    :param horizon_seconds: [s] time horizon being run.
    :param sampling_time: [s] sampling timestep.
    :return: dummy simple planner.
    """
    acceleration_np: npt.NDArray[np.float32] = np.asarray(acceleration)
    return SimplePlanner(
        horizon_seconds=horizon_seconds,
        sampling_time=sampling_time,
        acceleration=acceleration_np,
    )


def _create_dummy_simulation_history_buffer(
    scenario: AbstractScenario, iteration: int = 0, time_horizon: int = 2, num_samples: int = 2, buffer_size: int = 2
) -> SimulationHistoryBuffer:
    """
    Create dummy SimulationHistoryBuffer.
    :param scenario: Scenario.
    :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
    :param time_horizon: the desired horizon to the future.
    :param num_samples: number of entries in the future.
    :param buffer_size: size of buffer.
    :return: SimulationHistoryBuffer.
    """
    past_observation = list(
        scenario.get_past_tracked_objects(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
    )

    past_ego_states = list(
        scenario.get_ego_past_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
    )

    # Dummy history buffer
    history_buffer = SimulationHistoryBuffer.initialize_from_list(
        buffer_size=buffer_size,
        ego_states=past_ego_states,
        observations=past_observation,
        sample_interval=scenario.database_interval,
    )

    return history_buffer


def serialize_scenario(
    scenario: AbstractScenario, num_poses: int = 12, future_time_horizon: float = 6.0
) -> SimulationHistory:
    """
    Serialize a scenario to a list of scene dicts.
    :param scenario: Scenario.
    :param num_poses: Number of poses in trajectory.
    :param future_time_horizon: Future time horizon in trajectory.
    :return: SimulationHistory containing all scenes.
    """
    simulation_history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
    ego_controller = PerfectTrackingController(scenario)
    simulation_time_controller = StepSimulationTimeController(scenario)
    observations = TracksObservation(scenario)

    # Dummy history buffer
    history_buffer = _create_dummy_simulation_history_buffer(scenario=scenario)

    # Get all states
    for _ in range(simulation_time_controller.number_of_iterations()):
        iteration = simulation_time_controller.get_iteration()
        ego_state = ego_controller.get_state()
        observation = observations.get_observation()
        traffic_light_status = list(scenario.get_traffic_light_status_at_iteration(iteration.index))

        # Log play back trajectory
        current_state = scenario.get_ego_state_at_iteration(iteration.index)
        states = scenario.get_ego_future_trajectory(iteration.index, future_time_horizon, num_poses)
        trajectory = InterpolatedTrajectory(list(itertools.chain([current_state], states)))

        simulation_history.add_sample(
            SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status)
        )
        next_iteration = simulation_time_controller.next_iteration()

        if next_iteration:
            ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
            observations.update_observation(iteration, next_iteration, history_buffer)

    return simulation_history


def visualize_scenario(
    scenario: NuPlanScenario, save_dir: str = '/tmp/scenario_visualization/', bokeh_port: int = 8899
) -> None:
    """
    Visualize a scenario in Bokeh.
    :param scenario: Scenario object to be visualized.
    :param save_dir: Dir to save serialization and visualization artifacts.
    :param bokeh_port: Port that the server bokeh starts to render the generate the visualization will run on.
    """
    map_factory = NuPlanMapFactory(get_maps_db(map_root=scenario.map_root, map_version=scenario.map_version))

    simulation_history = serialize_scenario(scenario)
    simulation_scenario_key = save_scenes_to_dir(
        scenario=scenario, save_dir=save_dir, simulation_history=simulation_history
    )
    visualize_scenarios([simulation_scenario_key], map_factory, Path(save_dir), bokeh_port=bokeh_port)


def visualize_scenarios(
    simulation_scenario_keys: List[SimulationScenarioKey],
    map_factory: NuPlanMapFactory,
    save_path: Path,
    bokeh_port: int = 8899,
) -> None:
    """
    Visualize scenarios in Bokeh.
    :param simulation_scenario_keys: A list of simulation scenario keys.
    :param map_factory: Map factory object to use for rendering.
    :param save_path: Path where to save the scene dict.
    :param bokeh_port: Port that the server bokeh starts to render the generate the visualization will run on.
    """

    def complete_message() -> None:
        """Logging to print once the visualization is ready."""
        logger.info("Done rendering!")

    def notebook_url_callback(server_port: Optional[int]) -> str:
        """
        Callback that configures the bokeh server started by bokeh.io.show to accept requests
        from any origin. Without this, running a notebook on a port other than 8888 results in
        scenario visualizations not being rendered. For reference, see:
            - show() docs: https://docs.bokeh.org/en/latest/docs/reference/io.html#bokeh.io.show
            - downstream usage: https://github.com/bokeh/bokeh/blob/aae3034/src/bokeh/io/notebook.py#L545
        :param server_port: Passed by bokeh to indicate what port it started a server on (random by default).
        :return: Origin string and server url used by bokeh.
        """
        if server_port is None:
            return "*"
        return f"http://localhost:{server_port}"

    def bokeh_app(doc: Document) -> None:
        """
        Run bokeh app in jupyter notebook.
        :param doc: Bokeh document to render.
        """
        # Change simulation_main_path to a folder where you want to save rendered videos.
        nuboard_file = NuBoardFile(
            simulation_main_path=save_path.name,
            simulation_folder='',
            metric_main_path='',
            metric_folder='',
            aggregator_metric_folder='',
        )

        experiment_file_data = ExperimentFileData(file_paths=[nuboard_file])
        # Create a simulation tile
        simulation_tile = SimulationTile(
            doc=doc,
            map_factory=map_factory,
            experiment_file_data=experiment_file_data,
            vehicle_parameters=get_pacifica_parameters(),
        )

        # Render a simulation tile
        simulation_scenario_data = simulation_tile.render_simulation_tiles(simulation_scenario_keys)

        # Create layouts
        simulation_figures = [data.plot for data in simulation_scenario_data]
        simulation_layouts = column(simulation_figures)

        # Add the layouts to the bokeh document
        doc.add_root(simulation_layouts)
        doc.add_next_tick_callback(complete_message)

    # bokeh.io.show starts a server on `bokeh_port`, but doesn't return a handle to it. If it isn't
    # shut down, we get a port-in-use error when generating the new visualization. Thus, we search for
    # any server currently running on the assigned port and shut it down before calling `show` again.
    for server_uuid, server in curstate().uuid_to_server.items():
        if server.port == bokeh_port:
            server.unlisten()
            logging.debug("Shut down bokeh server %s running on port %d", server_uuid, bokeh_port)

    start_event_loop_if_needed()
    show(bokeh_app, notebook_url=notebook_url_callback, port=bokeh_port)


def get_default_scenario_extraction(
    scenario_duration: float = 15.0,
    extraction_offset: float = -2.0,
    subsample_ratio: float = 0.5,
) -> ScenarioExtractionInfo:
    """
    Get default scenario extraction instructions used in visualization.
    :param scenario_duration: [s] Duration of scenario.
    :param extraction_offset: [s] Offset of scenario (e.g. -2 means start scenario 2s before it starts).
    :param subsample_ratio: Scenario resolution.
    :return: Scenario extraction info object.
    """
    return ScenarioExtractionInfo(DEFAULT_SCENARIO_NAME, scenario_duration, extraction_offset, subsample_ratio)


def get_default_scenario_from_token(
    data_root: str, log_file_full_path: str, token: str, map_root: str, map_version: str
) -> NuPlanScenario:
    """
    Build a scenario with default parameters for visualization.
    :param data_root: The root directory to use for looking for db files.
    :param log_file_full_path: The full path to the log db file to use.
    :param token: Lidar pc token to be used as anchor for the scenario.
    :param map_root: The root directory to use for looking for maps.
    :param map_version: The map version to use.
    :return: Instantiated scenario object.
    """
    timestamp = get_sensor_data_token_timestamp_from_db(log_file_full_path, get_lidarpc_sensor_data(), token)
    map_name = get_sensor_token_map_name_from_db(log_file_full_path, get_lidarpc_sensor_data(), token)
    return NuPlanScenario(
        data_root=data_root,
        log_file_load_path=log_file_full_path,
        initial_lidar_token=token,
        initial_lidar_timestamp=timestamp,
        scenario_type=DEFAULT_SCENARIO_NAME,
        map_root=map_root,
        map_version=map_version,
        map_name=map_name,
        scenario_extraction_info=get_default_scenario_extraction(),
        ego_vehicle_parameters=get_pacifica_parameters(),
    )


def get_scenario_type_token_map(db_files: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get a map from scenario types to lists of all instances for a given scenario type in the database.
    :param db_files: db files to search for available scenario types.
    :return: dictionary mapping scenario type to list of db/token pairs of that type.
    """
    available_scenario_types = defaultdict(list)
    for db_file in db_files:
        for tag, token in get_lidarpc_tokens_with_scenario_tag_from_db(db_file):
            available_scenario_types[tag].append((db_file, token))

    return available_scenario_types


def visualize_nuplan_scenarios(
    data_root: str, db_files: str, map_root: str, map_version: str, bokeh_port: int = 8899
) -> None:
    """
    Create a dropdown box populated with unique scenario types to visualize from a database.
    :param data_root: The root directory to use for looking for db files.
    :param db_files: List of db files to load.
    :param map_root: The root directory to use for looking for maps.
    :param map_version: The map version to use.
    :param bokeh_port: Port that the server bokeh starts to render the generate the visualization will run on.
    """
    from IPython.display import clear_output, display
    from ipywidgets import Dropdown, Output

    log_db_files = discover_log_dbs(db_files)

    scenario_type_token_map = get_scenario_type_token_map(log_db_files)

    out = Output()
    drop_down = Dropdown(description='Scenario', options=sorted(scenario_type_token_map.keys()))

    def scenario_dropdown_handler(change: Any) -> None:
        """
        Dropdown handler that randomly chooses a scenario from the selected scenario type and renders it.
        :param change: Object containing scenario selection.
        """
        with out:
            clear_output()

            logger.info("Randomly rendering a scenario...")
            scenario_type = str(change.new)
            log_db_file, token = random.choice(scenario_type_token_map[scenario_type])
            scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)

            visualize_scenario(scenario, bokeh_port=bokeh_port)

    display(drop_down)
    display(out)
    drop_down.observe(scenario_dropdown_handler, names='value')


def setup_notebook() -> None:
    """
    Code that must be run at the start of every tutorial notebook to:
        - patch the event loop to allow nesting, eg. so we can run asyncio.run from
          within a notebook.
    """
    nest_asyncio.apply()


def start_event_loop_if_needed() -> None:
    """
    Starts event loop, if there isn't already one running.
    Should be called before funcitons that require the event loop to be running (or able
    to be auto-started) to work (eg. bokeh.show).
    """
    try:
        # Gets the running event loop, starting one in the main thread if none has
        # been created before.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If an event loop was already created and cleaned up, catch the runtime error
        # and create a new one.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
