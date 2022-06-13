import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from bokeh.palettes import Set1, Set2, Set3

from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
from nuplan.planning.nuboard.base.data_class import NuBoardFile, SimulationScenarioKey

logger = logging.getLogger(__name__)


@dataclass
class ExperimentFileData:
    """Data for experiment files."""

    file_paths: List[NuBoardFile]  # Experiment file path
    color_palettes: List[str] = field(default_factory=list)  # Color choices
    available_metric_statistics_names: List[str] = field(default_factory=list)  # Metric statistics name
    metric_statistics_dataframes: List[List[MetricStatisticsDataFrame]] = field(
        default_factory=list
    )  # Metric dataframe
    metric_aggregator_dataframes: List[Dict[str, pd.DataFrame]] = field(
        default_factory=list
    )  # Metric aggregator dataframe
    simulation_files: Dict[str, Any] = field(default_factory=dict)  # Simulation files
    simulation_scenario_keys: List[SimulationScenarioKey] = field(default_factory=list)  # Simulation scenario keys
    available_scenario_type_names: Dict[str, List[str]] = field(
        default_factory=dict
    )  # Available scenario types to scenario names in search
    available_scenario_types: List[str] = field(default_factory=list)  # Available scenario types in search
    available_scenario_log_names: Dict[str, List[str]] = field(
        default_factory=dict
    )  # Available log names to scenario names in search
    available_log_names: List[str] = field(default_factory=list)  # Available log names
    file_path_colors: Dict[int, Dict[str, str]] = field(default_factory=dict)  # Color for each experiment file
    color_index: int = 0  # Current color index

    def __post_init__(self) -> None:
        """Post initialization."""
        if not self.simulation_files:
            self.simulation_files = defaultdict(set)

        if not self.color_palettes:
            self.color_palettes = Set1[9] + Set2[8] + Set3[12]

        if self.file_paths:
            # Reset file paths
            file_paths = self.file_paths
            self.file_paths = []
            self.update_data(file_paths=file_paths)

    def update_data(self, file_paths: List[NuBoardFile]) -> None:
        """
        Update experiment data with a new list of nuboard file paths.
        :param file_paths: A list of new nuboard file paths.
        """
        starting_file_path_index = len(self.file_paths)
        # Update file path color
        self._update_file_path_color(file_paths=file_paths, starting_file_path_index=starting_file_path_index)

        # Add new metric files
        self._add_metric_files(file_paths=file_paths, starting_file_path_index=starting_file_path_index)

        # Add new metric aggregator files
        self._add_metric_aggregator_files(file_paths=file_paths, starting_file_path_index=starting_file_path_index)

        # Add new simulation files
        self._add_simulation_files(file_paths=file_paths, starting_file_path_index=starting_file_path_index)

        # Add to file paths
        self.file_paths += file_paths

    @staticmethod
    def _get_base_path(current_path: Path, base_path: Path, sub_folder: str) -> Path:
        """
        Get valid base path.
        :param current_path: Current nuboard file path.
        :Param base_path: Alternative base path.
        :param sub_folder: Sub folder.
        :return A base path.
        """
        default_path = base_path / sub_folder
        if current_path is None:
            return default_path

        base_folder = current_path / sub_folder
        if not base_folder.exists():
            base_folder = default_path
        return base_folder

    def _update_file_path_color(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Update file path colors.
        :param file_paths: A list of new nuboard file paths.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            self.file_path_colors[file_path_index] = defaultdict(str)
            metric_path = self._get_base_path(
                current_path=file_path.current_path,
                base_path=Path(file_path.metric_main_path),
                sub_folder=file_path.metric_folder,
            )
            if not metric_path.exists():
                continue

            planner_names: List[str] = []
            # Loop through metric parquet files
            for file in metric_path.iterdir():
                try:
                    data_frame = MetricStatisticsDataFrame.load_parquet(file)
                    planner_names += data_frame.planner_names
                except (FileNotFoundError, Exception):
                    # Ignore the file
                    pass

            # Remove duplicate planner names
            planner_names = list(set(planner_names))
            for planner_name in planner_names:
                self.file_path_colors[file_path_index][planner_name] = self.color_palettes[self.color_index]
                self.color_index += 1

    def _add_metric_files(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Add and load metric files.
        Folder hierarchy: planner_name -> scenario_type -> metric result name -> scenario_name.pkl
        :param file_paths: A list of new nuboard files.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            self.metric_statistics_dataframes.append([])
            metric_path = self._get_base_path(
                current_path=file_path.current_path,
                base_path=Path(file_path.metric_main_path),
                sub_folder=file_path.metric_folder,
            )
            if not metric_path.exists():
                continue

            # Loop through metric parquet files
            for file in metric_path.iterdir():
                try:
                    data_frame = MetricStatisticsDataFrame.load_parquet(file)
                    self.metric_statistics_dataframes[file_path_index].append(data_frame)
                    self.available_metric_statistics_names.append(data_frame.metric_statistic_name)
                except (FileNotFoundError, Exception):
                    # Ignore the file
                    pass

        # Remove duplicates
        self.available_metric_statistics_names = sorted(
            list(set(self.available_metric_statistics_names)), reverse=False
        )

    def _add_metric_aggregator_files(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Load metric aggregator files.
        :param file_paths: A list of new nuboard files.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            self.metric_aggregator_dataframes.append({})
            metric_aggregator_path = self._get_base_path(
                current_path=file_path.current_path,
                base_path=Path(file_path.metric_main_path),
                sub_folder=file_path.aggregator_metric_folder,
            )
            if not metric_aggregator_path.exists():
                continue
            # Loop through metric parquet files
            for file in metric_aggregator_path.iterdir():
                try:
                    data_frame = pd.read_parquet(file)
                    self.metric_aggregator_dataframes[file_path_index][file.stem] = data_frame
                except (FileNotFoundError, Exception):
                    # Ignore the file
                    pass

    def _add_simulation_files(self, file_paths: List[NuBoardFile], starting_file_path_index: int) -> None:
        """
        Load simulation files.
        Folder hierarchy: planner_name -> scenario_type -> scenario_names -> iteration.pkl.
        :param file_paths: A list of new nuboard files.
        :param starting_file_path_index: Starting file path index.
        """
        for index, file_path in enumerate(file_paths):
            file_path_index = starting_file_path_index + index
            simulation_path = self._get_base_path(
                current_path=file_path.current_path,
                base_path=Path(file_path.simulation_main_path),
                sub_folder=file_path.simulation_folder,
            )
            if not simulation_path.exists():
                continue
            planner_name_paths = simulation_path.iterdir()
            for planner_name_path in planner_name_paths:
                planner_name = planner_name_path.name
                scenario_type_paths = planner_name_path.iterdir()
                for scenario_type_path in scenario_type_paths:
                    log_name_paths = scenario_type_path.iterdir()
                    scenario_type = scenario_type_path.name
                    if scenario_type not in self.available_scenario_type_names:
                        self.available_scenario_type_names[scenario_type] = []
                    for log_name_path in log_name_paths:
                        scenario_name_paths = log_name_path.iterdir()
                        log_name = log_name_path.name
                        if log_name not in self.available_scenario_log_names:
                            self.available_scenario_log_names[log_name] = []
                        if log_name not in self.available_scenario_type_names[scenario_type]:
                            self.available_scenario_type_names[scenario_type].append(log_name)
                        for scenario_name_path in scenario_name_paths:
                            scenario_name = scenario_name_path.name
                            scenario_key = (
                                f"{simulation_path.parents[0].name}/{planner_name}/"
                                f"{scenario_type}/{log_name}/{scenario_name}"
                            )
                            if scenario_key in self.simulation_files:
                                continue
                            if scenario_name not in self.available_scenario_log_names[log_name]:
                                self.available_scenario_log_names[log_name].append(scenario_name)
                            files = scenario_name_path.iterdir()
                            for file in files:
                                self.simulation_files[scenario_key].add(file)
                            self.simulation_scenario_keys.append(
                                SimulationScenarioKey(
                                    nuboard_file_index=file_path_index,
                                    log_name=log_name,
                                    planner_name=planner_name,
                                    scenario_type=scenario_type,
                                    scenario_name=scenario_name,
                                    files=list(self.simulation_files[scenario_key]),
                                )
                            )

        # Add scenario types
        available_scenario_types = list(set(self.available_scenario_type_names.keys()))
        self.available_scenario_types = sorted(available_scenario_types, reverse=False)

        # Add log names
        available_log_names = list(set(self.available_scenario_log_names.keys()))
        self.available_log_names = sorted(available_log_names, reverse=False)
