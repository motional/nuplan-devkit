from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass
from typing import Dict, List, NamedTuple


class BokehAgentStates(NamedTuple):
    """ Agent states in bokeh. """

    xs: List[List[List[List[float]]]]   # [m], [[list of [[Polygon connected corners in x]]]]
    ys: List[List[List[List[float]]]]   # [m], [[list of [[Polygon connected corners in y]]]]
    agent_type: List[str]   # Agent's category
    track_id: List[int]     # Agent's track id
    trajectory_x: List[List[float]]  # [m], [a list of trajectory in x]
    trajectory_y: List[List[float]]     # [m], [a list of trajectory in y]


@dataclass
class MetricScenarioKey:
    """ Metric key for scenario in nuBoard. """

    planner_name: str
    scenario_type: str
    scenario_name: str
    metric_result_name: str
    file: pathlib.Path


@dataclass
class SimulationScenarioKey:
    """ Simulation key for scenario in nuBoard. """

    planner_name: str
    scenario_type: str
    scenario_name: str
    files: List[pathlib.Path]


@dataclass
class NuBoardFile:
    """ Data class to save nuBoard file info. """

    main_path: str
    simulation_folder: str
    metric_folder: str

    @classmethod
    def extension(cls) -> str:
        return '.nuboard'

    def __eq__(self, other: object) -> bool:
        """
        Comparison between two NuBoardFile.
        :param other: Other object.
        :return True if both objects are same.
        """

        if not isinstance(other, NuBoardFile):
            raise NotImplementedError

        return other.main_path == self.main_path and other.simulation_folder == self.simulation_folder and \
            other.metric_folder == self.metric_folder

    def save_nuboard_file(self, file: pathlib.Path) -> None:
        """
        Save NuBoardFile data class to a file.
        :param file: The saved file path.
        """

        with open(file, 'wb') as f:
            pickle.dump(self.serialize(), f)

    @classmethod
    def load_nuboard_file(cls, file: pathlib.Path) -> NuBoardFile:
        """
        Read a NuBoard file to NuBoardFile data class.
        :file: NuBoard file path.
        """

        with open(file, 'rb') as f:
            data = pickle.load(f)

        return cls.deserialize(data=data)

    def serialize(self) -> Dict[str, str]:
        """
        Serialization of NuBoardFile data class to dictionary.
        :return A serialized dictionary class.
        """

        return {
            'main_path': self.main_path,
            'simulation_folder': self.simulation_folder,
            'metric_folder': self.metric_folder
        }

    @classmethod
    def deserialize(cls, data: Dict[str, str]) -> NuBoardFile:
        """
        Deserialization of a NuBoard file into NuBoardFile dtata class.
        :param data: A serialized nuboard file data.
        :return A NuBoard file data class.
        """

        main_path = data['main_path'].replace('//', '/')
        return NuBoardFile(main_path=main_path,
                           simulation_folder=data['simulation_folder'],
                           metric_folder=data['metric_folder']
                           )
