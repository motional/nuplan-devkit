import logging
import lzma
import pathlib
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import msgpack
import ujson as json
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.database.utils.label.utils import PBVTB_LABELMAPPING
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.utils.serialization.to_scene import create_trajectory_structure, \
    to_scene_agent_prediction_from_boxes, to_scene_boxes, to_scene_ego_from_center_pose

logger = logging.getLogger(__name__)


@dataclass
class SceneColors:
    ego_predicted_trajectory = [0, 0, 255, 100]  # [r, g, b, a] color
    ego_expert_trajectory = [255, 0, 0, 100]  # [r, g, b, a] color
    agents_predicted_trajectory = [0, 255, 0, 50]  # [r, g, b, a] color


def _dump_to_json(file: pathlib.Path, scene_to_save: Any) -> None:
    """ Dump file into json """
    with open(str(file.with_suffix(".json")), 'w') as f:
        json.dump(scene_to_save, f)


def _dump_to_pickle(file: pathlib.Path, scene_to_save: Any) -> None:
    """ Dump file into compressed pickle """
    with lzma.open(file.with_suffix(".pkl.xz"), "wb") as f:  # type: ignore
        pickle.dump(scene_to_save, f)  # type: ignore


def _dump_to_msgpack(file: pathlib.Path, scene_to_save: Any) -> None:
    """ Dump file into compressed msgpack """
    with lzma.open(file.with_suffix(".msgpack.xz"), "wb") as f:  # type: ignore
        f.write(msgpack.packb(scene_to_save))


def _dump_to_file(file: pathlib.Path, scene_to_save: Any, serialization_type: str) -> None:
    """
    Dump scene into file
    :param serialization_type: type of serialization ["json", "pickle", "msgpack"]
    :param file: file name
    :param scene_to_save: what to store
    """
    if serialization_type == "json":
        _dump_to_json(file, scene_to_save)
    elif serialization_type == "pickle":
        _dump_to_pickle(file, scene_to_save)
    elif serialization_type == "msgpack":
        _dump_to_msgpack(file, scene_to_save)
    else:
        raise ValueError(f"Unknown option: {serialization_type}")


def convert_sample_to_scene(map_name: str,
                            traffic_light_status: List[TrafficLightStatusData],
                            mission_goal: Optional[StateSE2],
                            expert_trajectory: List[EgoState],
                            data: SimulationHistorySample,
                            vehicle: VehicleParameters,
                            colors: SceneColors = SceneColors()) -> Dict[str, Any]:
    """
    Serialize history and scenario.
    :param map_name: name of the map used for this scenario.
    :param traffic_light_status: Traffic light status.
    :param mission_goal: if mission goal is present, this is goal of this mission.
    :param expert_trajectory: trajectory of an expert driver.
    :param data: single sample from history.
    :param vehicle: vehicle parameters.
    :param colors: colors for trajectories.
    :return: serialized dictionary.
    """

    # Initialize scene
    scene: Dict[str, Any] = {"timestamp_us": data.ego_state.time_us}
    trajectories: Dict[str, Dict[str, Any]] = {}

    # Convert goal
    if mission_goal is not None:
        scene["goal"] = to_scene_ego_from_center_pose(mission_goal, vehicle)
    else:
        scene["goal"] = None

    # Convert ego pose
    scene["ego"] = to_scene_ego_from_center_pose(data.ego_state.rear_axle, vehicle)
    scene["ego"]["timestamp_us"] = data.ego_state.time_us

    # Convert Map Area
    map_name_without_suffix = str(pathlib.Path(map_name).with_suffix(""))
    scene["map"] = {"area": map_name_without_suffix}
    scene["map_name"] = map_name

    # Convert Detections
    if isinstance(data.observation, Detections):
        scene["world"] = to_scene_boxes(data.observation.boxes, PBVTB_LABELMAPPING)
        scene["prediction"] = to_scene_agent_prediction_from_boxes(data.observation.boxes,
                                                                   colors.agents_predicted_trajectory)

    # Convert Trajectory
    trajectories["ego_predicted_trajectory"] = create_trajectory_structure(
        data.trajectory.get_sampled_trajectory(),
        colors.ego_predicted_trajectory)

    # Convert Scenario
    trajectories["ego_expert_trajectory"] = create_trajectory_structure(expert_trajectory, colors.ego_expert_trajectory)

    # Store trajectories
    scene["trajectories"] = trajectories

    # Serialize traffic light status
    scene["traffic_light_status"] = [traffic_light.serialize() for traffic_light in traffic_light_status]
    return scene


class SerializationCallback(AbstractCallback):

    def __init__(self,
                 output_directory: Union[str, pathlib.Path],
                 folder_name: Union[str, pathlib.Path],
                 serialization_type: str,
                 serialize_into_single_file: bool,
                 vehicle: VehicleParameters = get_pacifica_parameters()):
        """
        Construct serialization callback
        :param output_directory: where scenes should be serialized
        :param folder_name: folder where output should be serialized
        :param serialize_into_single_file: if true all data will be in single file, if false, each time step will
                be serialized into a separate file
        :param serialization_type: A way to serialize output, options: ["json", "pickle", "msgpack"]
        """
        available_formats = ["json", "pickle", "msgpack"]
        if serialization_type not in available_formats:
            raise ValueError("The serialization callback will not store files anywhere!"
                             f"Choose at least one format from {available_formats} instead of {serialization_type}!")

        self._output_directory = pathlib.Path(output_directory) / folder_name
        self._serialization_type = serialization_type
        self._vehicle = vehicle
        self._serialize_into_single_file = serialize_into_single_file

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Create directory at initialization
        :param setup: simulation setup
        :param planner: planner before initialization
        """
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        scenario_directory.mkdir(exist_ok=True, parents=True)

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """
        On reached_end validate that all steps were correctly serialized
        :param setup: simulation setup
        :param planner: planner when simulation ends
        :param history: resulting from simulation
        """
        number_of_scenes = len(history)
        if number_of_scenes == 0:
            raise RuntimeError("Number of scenes has to be greater than 0")

        # Create directory
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        scenes = [self._serialize_history_sample(setup.scenario, sample,
                                                 setup.scenario.get_traffic_light_status_at_iteration(index))
                  for index, sample in enumerate(history.data)]

        # Serialize based on preference
        self._serialize_scenes(scenes, scenario_directory)

    def _serialize_scenes(self, scenes: List[Dict[str, Any]], scenario_directory: pathlib.Path) -> None:
        """
        Serialize scenes based on callback setup to json/pickle or other
        :param scenes: scenes to be serialized
        :param scenario_directory: directory where they should be serialized
        """

        if not self._serialize_into_single_file:
            # Split data into many smaller files
            for scene in scenes:
                file_name = (scenario_directory / str(scene["ego"]["timestamp_us"]))
                _dump_to_file(file_name, scene, self._serialization_type)
        else:
            # Dump all data into a single file
            file_name = (scenario_directory / scenario_directory.name)
            _dump_to_file(file_name, scenes, self._serialization_type)

    def _get_scenario_folder(self, planner_name: str, scenario: AbstractScenario) -> pathlib.Path:
        """
        Compute scenario folder directory where all files will be stored
        :param planner_name: planner name
        :param scenario: for which to compute directory name
        :return directory path
        """
        return self._output_directory / planner_name / scenario.scenario_type / scenario.scenario_name  # type: ignore

    def _serialize_history_sample(self,
                                  scenario: AbstractScenario,
                                  data: SimulationHistorySample,
                                  traffic_light_status: List[TrafficLightStatusData],
                                  colors: SceneColors = SceneColors()) -> Dict[str, Any]:
        """
        Serialize history and scenario
        :param scenario: input scenario
        :param data: single sample from history
        :param colors: colors for trajectories
        :return serialized dictionary
        """
        return convert_sample_to_scene(map_name=scenario.map_api.map_name,
                                       traffic_light_status=traffic_light_status,
                                       expert_trajectory=scenario.get_expert_ego_trajectory(),
                                       mission_goal=scenario.get_mission_goal(),
                                       data=data, colors=colors, vehicle=self._vehicle)
