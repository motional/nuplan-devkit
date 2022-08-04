from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple, Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation, Sensors


class SimulationHistoryBuffer:
    """
    This class is used to keep a rolling buffer of a given size. The buffer is a first-in first-out queue. Hence, the
    oldest samples in the buffer are continuously replaced as new samples are appended.
    """

    def __init__(
        self,
        ego_state_buffer: Deque[EgoState],
        observations_buffer: Deque[Observation],
        sample_interval: Optional[float] = None,
    ):
        """
        Constructs a SimulationHistoryBuffer
        :param ego_state_buffer: Past ego state trajectory including the state.
            at the current time step [t_-N, ..., t_-1, t_0]
        :param observations_buffer: Past observations including the observation.
            at the current time step [t_-N, ..., t_-1, t_0].
        :param sample_interval: [s] the time interval between each sample, if given
        """
        if not ego_state_buffer or not observations_buffer:
            raise ValueError('Ego and observation buffers cannot be empty!')

        if len(ego_state_buffer) != len(observations_buffer):
            raise ValueError(
                'Ego and observations buffer is '
                f'not the same length {len(ego_state_buffer) != len(observations_buffer)}!'
            )

        self._ego_state_buffer = ego_state_buffer
        self._observations_buffer = observations_buffer
        self._sample_interval = sample_interval

    @property
    def ego_state_buffer(self) -> Deque[EgoState]:
        """
        :return: current ego state buffer
        """
        return self._ego_state_buffer

    @property
    def observation_buffer(self) -> Deque[Observation]:
        """
        :return: current observation buffer
        """
        return self._observations_buffer

    @property
    def size(self) -> int:
        """
        :return: Size of the buffer.
        """
        return len(self.ego_states)

    @property
    def duration(self) -> Optional[float]:
        """
        :return: [s] Duration of the buffer.
        """
        return self.sample_interval * self.size if self.sample_interval else None

    @property
    def current_state(self) -> Tuple[EgoState, Observation]:
        """
        :return: current state of AV vehicle and its observations
        """
        return self.ego_states[-1], self.observations[-1]

    @property
    def sample_interval(self) -> Optional[float]:
        """
        :return: the sample interval
        """
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, sample_interval: float) -> None:
        """
        Sets the sample interval of the buffer, raises if the sample interval was not None
        :param sample_interval: The sample interval of the buffer
        """
        assert self._sample_interval is None, "Can't overwrite a pre-existing sample-interval!"
        self._sample_interval = sample_interval

    @property
    def ego_states(self) -> List[EgoState]:
        """
        :return: the ego state buffer in increasing temporal order where the last sample is the more recent sample
                 [t_-N, ..., t_-1, t_0]
        """
        return list(self._ego_state_buffer)

    @property
    def observations(self) -> List[Observation]:
        """
        :return: the observation buffer in increasing temporal order where the last sample is the more recent sample
                 [t_-N, ..., t_-1, t_0]
        """
        return list(self._observations_buffer)

    def append(self, ego_state: EgoState, observation: Observation) -> None:
        """
        Adds new samples to the buffers
        :param ego_state: an ego state
        :param observation: an observation
        """
        self._ego_state_buffer.append(ego_state)
        self._observations_buffer.append(observation)

    def extend(self, ego_states: List[EgoState], observations: List[Observation]) -> None:
        """
        Adds new samples to the buffers
        :param ego_states: an ego states list
        :param observations: an observations list
        """
        if len(ego_states) != len(observations):
            raise ValueError(f'Ego and observations are not the same length {len(ego_states) != len(observations)}!')
        self._ego_state_buffer.extend(ego_states)
        self._observations_buffer.extend(observations)

    def __len__(self) -> int:
        """
        :return: the length of the buffer
        @raise AssertionError if the length of each buffers are not the same
        """
        return len(self._ego_state_buffer)

    @classmethod
    def initialize_from_list(
        cls,
        buffer_size: int,
        ego_states: List[EgoState],
        observations: List[Observation],
        sample_interval: Optional[float] = None,
    ) -> SimulationHistoryBuffer:
        """
        Create history buffer from lists
        :param buffer_size: size of buffer
        :param ego_states: list of ego states
        :param observations: list of observations
        :param sample_interval: [s] the time interval between each sample, if given
        :return: SimulationHistoryBuffer
        """
        ego_state_buffer: Deque[EgoState] = deque(ego_states[-buffer_size:], maxlen=buffer_size)
        observations_buffer: Deque[Observation] = deque(observations[-buffer_size:], maxlen=buffer_size)

        return cls(
            ego_state_buffer=ego_state_buffer, observations_buffer=observations_buffer, sample_interval=sample_interval
        )

    @staticmethod
    def initialize_from_scenario(
        buffer_size: int, scenario: AbstractScenario, observation_type: Type[Observation]
    ) -> SimulationHistoryBuffer:
        """
        Initializes ego_state_buffer and observations_buffer from scenario
        :param buffer_size: size of the buffer
        :param scenario: Simulation scenario
        :param observation_type: Observation type used for the simulation
        """
        buffer_duration = buffer_size * scenario.database_interval

        if observation_type == DetectionsTracks:
            observation_getter = scenario.get_past_tracked_objects
        elif observation_type == Sensors:
            observation_getter = scenario.get_past_sensors
        else:
            raise ValueError(f"No matching observation type for {observation_type} for history!")

        past_observation = list(observation_getter(iteration=0, time_horizon=buffer_duration, num_samples=buffer_size))

        past_ego_states = list(
            scenario.get_ego_past_trajectory(iteration=0, time_horizon=buffer_duration, num_samples=buffer_size)
        )

        return SimulationHistoryBuffer.initialize_from_list(
            buffer_size=buffer_size,
            ego_states=past_ego_states,
            observations=past_observation,
            sample_interval=scenario.database_interval,
        )
