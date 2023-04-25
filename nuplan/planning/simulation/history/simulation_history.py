from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


@dataclass(frozen=True)
class SimulationHistorySample:
    """
    Single SimulationHistory sample point.
    """

    iteration: SimulationIteration  # The simulation iteration the sample was appended
    ego_state: EgoState  # The ego state
    trajectory: AbstractTrajectory  # The ego planned trajectory
    observation: Observation  # The observations (vehicles, pedestrians, cyclists)
    traffic_light_status: List[TrafficLightStatusData]  # Traffic light status


class SimulationHistory:
    """
    Simulation history including a sequence of simulation states.
    """

    def __init__(
        self, map_api: AbstractMap, mission_goal: StateSE2, data: Optional[List[SimulationHistorySample]] = None
    ) -> None:
        """
        Construct the history
        :param map_api: abstract map api for accessing the maps
        :param mission_goal: mission goal for which this history was recorded for
        :param data: A list of simulation data.
        """
        self.map_api: AbstractMap = map_api
        self.mission_goal = mission_goal

        self.data: List[SimulationHistorySample] = data if data is not None else list()

    def add_sample(self, sample: SimulationHistorySample) -> None:
        """
        Add a sample to history
        :param sample: one snapshot of a simulation
        """
        self.data.append(sample)

    def last(self) -> SimulationHistorySample:
        """
        :return: last sample from history, or raise if empty
        """
        if not self.data:
            raise RuntimeError("Data is empty!")
        return self.data[-1]

    def reset(self) -> None:
        """
        Clear the stored data
        """
        self.data.clear()

    def __len__(self) -> int:
        """
        Return the number of history samples as len().
        """
        return len(self.data)

    @property
    def extract_ego_state(self) -> List[EgoState]:
        """
        Extract ego states in simulation history.
        :return An List of ego_states.
        """
        return [sample.ego_state for sample in self.data]

    @property
    def interval_seconds(self) -> float:
        """
        Return the interval between SimulationHistorySamples.
        :return The interval in seconds.
        """
        if not self.data or len(self.data) < 1:
            raise ValueError("Data is empty!")
        elif len(self.data) < 2:
            raise ValueError("Can't calculate the interval of a single-iteration simulation.")

        return float(self.data[1].iteration.time_s - self.data[0].iteration.time_s)  # float cast is for mypy
