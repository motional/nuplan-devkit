import logging
import pathlib
from typing import Union

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.simulation_setup import SimulationSetup

logger = logging.getLogger(__name__)


class SimulationLogCallback(AbstractCallback):
    """
    Callback for simulation logging/object serialization to disk.
    """

    def __init__(
        self,
        output_directory: Union[str, pathlib.Path],
        simulation_log_dir: Union[str, pathlib.Path],
        serialization_type: str,
    ):
        """
        Construct simulation log callback.
        :param output_directory: where scenes should be serialized.
        :param simulation_log_dir: Folder where to save simulation logs.
        :param serialization_type: A way to serialize output, options: ["json", "pickle", "msgpack"]
        """
        available_formats = ["pickle", "msgpack"]
        if serialization_type not in available_formats:
            raise ValueError(
                "The simulation log callback will not store files anywhere!"
                f"Choose at least one format from {available_formats} instead of {serialization_type}!"
            )

        self._output_directory = pathlib.Path(output_directory) / simulation_log_dir
        self._serialization_type = serialization_type
        if serialization_type == "pickle":
            file_suffix = '.pkl.xz'
        elif serialization_type == "msgpack":
            file_suffix = '.msgpack.xz'
        else:
            raise ValueError(f"Unknown option: {serialization_type}")
        self._file_suffix = file_suffix

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

        scenario = setup.scenario
        file_name = scenario_directory / scenario.scenario_name
        file_name = file_name.with_suffix(self._file_suffix)

        simulation_log = SimulationLog(
            file_path=file_name, scenario=scenario, planner=planner, simulation_history=history
        )

        simulation_log.save_to_file()

    def _get_scenario_folder(self, planner_name: str, scenario: AbstractScenario) -> pathlib.Path:
        """
        Compute scenario folder directory where all files will be stored
        :param planner_name: planner name
        :param scenario: for which to compute directory name
        :return directory path
        """
        return self._output_directory / planner_name / scenario.scenario_type  # type: ignore
