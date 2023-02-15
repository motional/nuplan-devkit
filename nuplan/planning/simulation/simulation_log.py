from __future__ import annotations

import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack

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
        """Dump file into compressed pickle"""
        with lzma.open(self.file_path, "wb", preset=0) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _dump_to_msgpack(self) -> None:
        """Dump file into compressed msgpack"""
        # Serialize to a pickle object
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        with lzma.open(self.file_path, "wb", preset=0) as f:
            f.write(msgpack.packb(pickle_object))

    def save_to_file(self) -> None:
        """Dump simulation log into file."""
        serialization_type = self.simulation_log_type(self.file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if serialization_type == "pickle":
            self._dump_to_pickle()
        elif serialization_type == "msgpack":
            self._dump_to_msgpack()
        else:
            raise ValueError(f"Unknown option: {serialization_type}")

    @staticmethod
    def simulation_log_type(file_path: Path) -> str:
        """
        Deduce the simulation log type.
        :param file_path: File path.
        :return: one from ["msgpack", "pickle", "json"].
        """
        msg_pack = file_path.suffixes == ['.msgpack', '.xz']
        msg_pickle = file_path.suffixes == ['.pkl', '.xz']
        number_of_available_types = int(msg_pack) + int(msg_pickle)

        # We can handle only conclusive serialization type
        if number_of_available_types != 1:
            raise RuntimeError(f"Inconclusive file type: {file_path}!")

        if msg_pickle:
            return "pickle"
        elif msg_pack:
            return "msgpack"
        else:
            raise RuntimeError("Unknown condition!")

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
