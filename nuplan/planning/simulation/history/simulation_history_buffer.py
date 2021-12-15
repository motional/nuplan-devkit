from collections import deque
from typing import Deque, List, Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import Detections, Observation, Sensors


class SimulationHistoryBuffer:
    """
    This class is used to keep a rolling buffer of a given size. The buffer is a first-in first-out queue. Hence, the
    oldest samples in the buffer are continuously replaced as new samples are appended
    """

    def __init__(self, buffer_size: int):
        """
        Constructs a SimulationHistoryBuffer
        :param buffer_size: size of the buffer
        """
        self._buffer_size = buffer_size
        self._ego_state_buffer: Deque[EgoState] = deque(maxlen=buffer_size)
        self._observations_buffer: Deque[Observation] = deque(maxlen=buffer_size)

    def initialize(self, scenario: AbstractScenario, observation_type: Type[Observation]) -> None:
        """
        Initializes ego_state_buffer and observations_buffer from scenario
        :param scenario: Simulation scenario
        :param observation_type: Observation type used for the simulation
        """
        buffer_duration = self._buffer_size * scenario.database_interval

        if observation_type == Detections:
            past_observation = scenario.get_past_detections(iteration=0, time_horizon=buffer_duration,
                                                            num_samples=self._buffer_size)
        elif observation_type == Sensors:
            past_observation = scenario.get_past_sensors(iteration=0, time_horizon=buffer_duration,
                                                         num_samples=self._buffer_size)
        else:
            raise ValueError(f"No matching observation type for {observation_type} for history!")

        past_ego_states = scenario.get_ego_past_trajectory(iteration=0, time_horizon=buffer_duration,
                                                           num_samples=self._buffer_size)

        for ego_state, observation in zip(past_ego_states, past_observation):
            self._ego_state_buffer.append(ego_state)
            self._observations_buffer.append(observation)

        assert len(self._ego_state_buffer) == len(self._observations_buffer), \
            f"Simulation history queues length must match. Got ego_state_queue length {len(self._ego_state_buffer)} " \
            f"and observations_queue length {len(self._observations_buffer)}"

    @property
    def ego_states(self) -> List[EgoState]:
        """
        :return: the ego state buffer in increasing temporal order where the last sample is the more recent sample
                 [t_-N, ..., t_-1, t_0]
        """
        if len(self._ego_state_buffer) < 1:
            raise AttributeError("Simulation history buffer is empty")

        return list(self._ego_state_buffer)

    @property
    def observations(self) -> List[Observation]:
        """
        :return: the observation buffer in increasing temporal order where the last sample is the more recent sample
                 [t_-N, ..., t_-1, t_0]
        """
        if len(self._observations_buffer) < 1:
            raise AttributeError("Simulation history buffer is empty")

        return list(self._observations_buffer)

    def append(self, ego_state: EgoState, observation: Observation) -> None:
        """
        Adds new samples to the buffers
        :param ego_state: an ego state
        :param observation: an observation
        """
        self._ego_state_buffer.append(ego_state)
        self._observations_buffer.append(observation)

    def __len__(self) -> int:
        """
        :return: the length of the buffer
        @raise AssertionError if the length of each buffers are not the same
        """
        assert len(self._ego_state_buffer) == len(self._observations_buffer), \
            f"Simulation history queues length must match. Got ego_state_queue length {len(self._ego_state_buffer)} " \
            f"and observations_queue length {len(self._observations_buffer)}"

        return len(self._ego_state_buffer)
