from __future__ import annotations

import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack

from nuplan.common.utils.io_utils import save_buffer
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner


@dataclass
class SimulationLog:
    """Simulation log."""

    file_path: Path
    scenario: AbstractScenario
    planner: AbstractPlanner
    simulation_history: SimulationHistory

    def _dump_to_pickle(self) -> None:
        """
        Dump file into compressed pickle.
        """
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        save_buffer(self.file_path, lzma.compress(pickle_object, preset=0))

    def _dump_to_msgpack(self) -> None:
        """
        Dump file into compressed msgpack.
        """
        # Serialize to a pickle object
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        msg_packed_bytes = msgpack.packb(pickle_object)
        save_buffer(self.file_path, lzma.compress(msg_packed_bytes, preset=0))

    def save_to_file(self) -> None:
        """
        Dump simulation log into file.
        """
        serialization_type = self.simulation_log_type(self.file_path)

        if serialization_type == "pickle":
            self._dump_to_pickle()
        elif serialization_type == "msgpack":
            self._dump_to_msgpack()
        else:
            raise ValueError(f"Unknown option: {serialization_type}")

    @staticmethod
    def simulation_log_type(file_path: Path) -> str:
        """
        Deduce the simulation log type based on the last two portions of the suffix.
        The last suffix must be .xz, since we always dump/load to/from an xz container.
        If the second to last suffix is ".msgpack", assumes the log is of type "msgpack".
        If the second to last suffix is ".pkl", assumes the log is of type "pickle."
        If it's neither, raises a ValueError.
        Examples:
        - "/foo/bar/baz.1.2.pkl.xz" -> "pickle"
        - "/foo/bar/baz/1.2.msgpack.xz" -> "msgpack"
        - "/foo/bar/baz/1.2.msgpack.pkl.xz" -> "pickle"
        - "/foo/bar/baz/1.2.msgpack" -> Error
        :param file_path: File path.
        :return: one from ["msgpack", "pickle"].
        """
        # Make sure we have at least 2 suffixes
        if len(file_path.suffixes) < 2:
            raise ValueError(f"Inconclusive file type: {file_path}")

        # Assert last suffix is .xz
        last_suffix = file_path.suffixes[-1]
        if last_suffix != ".xz":
            raise ValueError(f"Inconclusive file type: {file_path}")

        # Assert we can deduce the type
        second_to_last_suffix = file_path.suffixes[-2]
        log_type_mapping = {
            ".msgpack": "msgpack",
            ".pkl": "pickle",
        }
        if second_to_last_suffix not in log_type_mapping:
            raise ValueError(f"Inconclusive file type: {file_path}")

        return log_type_mapping[second_to_last_suffix]

    @classmethod
    def load_data(cls, file_path: Path) -> Any:
        """Load simulation log."""
        simulation_log_type = SimulationLog.simulation_log_type(file_path=file_path)
        if simulation_log_type == "msgpack":
            with lzma.open(str(file_path), "rb") as f:
                data = msgpack.unpackb(f.read())

                # pickle load
                data = pickle.loads(data)

        elif simulation_log_type == "pickle":
            with lzma.open(str(file_path), "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unknown serialization type: {simulation_log_type}!")

        return data
