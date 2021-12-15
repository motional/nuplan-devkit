from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory


@dataclass(frozen=True)
class SimulationHistorySample:
    iteration: SimulationIteration  # The simulation iteration the sample was appended
    ego_state: EgoState           # The ego state
    trajectory: AbstractTrajectory  # The ego planned trajectory
    observation: Observation        # The observations (vehicles, pedestrians, cyclists)


class SimulationHistory:
    def __init__(self,
                 map_api: AbstractMap,
                 mission_goal: StateSE2) -> None:
        """
        Construct the history
        :param map_api: abstract map api for accessing the maps
        :param mission_goal: mission goal for which this history was recorded for
        """
        self.map_api: AbstractMap = map_api
        self.mission_goal = mission_goal
        self.data: List[SimulationHistorySample] = list()

    def add_sample(self, sample: SimulationHistorySample) -> None:
        """
        Add a sample to history
        :param sample: one snapshot of a simulation
        """
        self.data.append(sample)

    def clear(self) -> None:
        """
        Clear the stored data
        """
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)
