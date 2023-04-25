from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

from nuplan.common.utils.io_utils import save_object_as_pickle
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame


@dataclass
class MetricScenarioKey:
    """Metric key for scenario in nuBoard."""

    log_name: str
    planner_name: str
    scenario_type: str
    scenario_name: str
    metric_result_name: str
    file: pathlib.Path


@dataclass
class SimulationScenarioKey:
    """Simulation key for scenario in nuBoard."""

    log_name: str
    planner_name: str
    scenario_type: str
    scenario_name: str
    files: List[pathlib.Path]
    nuboard_file_index: int


@dataclass
class NuBoardFile:
    """Data class to save nuBoard file info."""

    simulation_main_path: str  # Simulation main path
    metric_main_path: str  # Metric main path
    metric_folder: str  # Metric folder
    aggregator_metric_folder: str  # Aggregated metric folder

    simulation_folder: Optional[str] = None  # Simulation folder, or None if the SimulationLog wasn't serialized
    current_path: Optional[pathlib.Path] = None  # Current path of the nuboard file

    @classmethod
    def extension(cls) -> str:
        """Return nuboard file extension."""
        return ".nuboard"

    def __eq__(self, other: object) -> bool:
        """
        Comparison between two NuBoardFile.
        :param other: Other object.
        :return True if both objects are same.
        """
        if not isinstance(other, NuBoardFile):
            return NotImplemented

        return (
            other.simulation_main_path == self.simulation_main_path
            and other.simulation_folder == self.simulation_folder
            and other.metric_main_path == self.metric_main_path
            and other.metric_folder == self.metric_folder
            and other.aggregator_metric_folder == self.aggregator_metric_folder
            and other.current_path == self.current_path
        )

    def save_nuboard_file(self, filename: pathlib.Path) -> None:
        """
        Save NuBoardFile data class to a file.
        :param filename: The saved file path.
        """
        save_object_as_pickle(filename, self.serialize())

    @classmethod
    def load_nuboard_file(cls, filename: pathlib.Path) -> NuBoardFile:
        """
        Read a NuBoard file to NuBoardFile data class.
        :file: NuBoard file path.
        """
        with open(filename, "rb") as file:
            data = pickle.load(file)

        return cls.deserialize(data=data)

    def serialize(self) -> Dict[str, str]:
        """
        Serialization of NuBoardFile data class to dictionary.
        :return A serialized dictionary class.
        """
        as_dict = {
            "simulation_main_path": self.simulation_main_path,
            "metric_main_path": self.metric_main_path,
            "metric_folder": self.metric_folder,
            "aggregator_metric_folder": self.aggregator_metric_folder,
        }

        if self.simulation_folder is not None:
            as_dict["simulation_folder"] = self.simulation_folder

        return as_dict

    @classmethod
    def deserialize(cls, data: Dict[str, str]) -> NuBoardFile:
        """
        Deserialization of a NuBoard file into NuBoardFile data class.
        :param data: A serialized nuboard file data.
        :return A NuBoard file data class.
        """
        simulation_main_path = data["simulation_main_path"].replace("//", "/")
        metric_main_path = data["metric_main_path"].replace("//", "/")
        return NuBoardFile(
            simulation_main_path=simulation_main_path,
            simulation_folder=data.get("simulation_folder", None),
            metric_main_path=metric_main_path,
            metric_folder=data["metric_folder"],
            aggregator_metric_folder=data["aggregator_metric_folder"],
        )


@dataclass
class SelectedMetricStatisticDataFrame:
    """
    Selected metric statistics dataframe
    """

    dataframe_index: int  # dataframe index
    dataframe: MetricStatisticsDataFrame  # metric statistics dataframe
