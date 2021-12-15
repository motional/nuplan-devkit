from functools import cached_property
from typing import Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import Detections, Observation
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_agent_manager import IDMAgentManager
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_agents_builder import \
    build_idm_agents_on_map_rails
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration


class IDMAgentsObservation(AbstractObservation):
    """
    Replay bounding boxes from samples.
    """

    def __init__(self, target_velocity: float,
                 min_gap_to_lead_agent: float,
                 headway_time: float,
                 accel_max: float,
                 decel_max: float,
                 scenario: AbstractScenario,
                 planned_trajectory_samples: int = 10,
                 planned_trajectory_sample_interval: float = 0.5):
        """
        Constructor for IDMAgentsObservation

        :param target_velocity: [m/s] Desired velocity in free traffic
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front
        :param accel_max: [m/s^2] maximum acceleration
        :param decel_max: [m/s^2] maximum deceleration (positive value)
        :param scenario: scenario
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        """
        self.current_iteration = 0

        self._target_velocity = target_velocity
        self._min_gap_to_lead_agent = min_gap_to_lead_agent
        self._headway_time = headway_time
        self._accel_max = accel_max
        self._decel_max = decel_max
        self._scenario = scenario
        self._planned_trajectory_samples = planned_trajectory_samples
        self._planned_trajectory_sample_interval = planned_trajectory_sample_interval

    @cached_property
    def _idm_agent_manager(self) -> IDMAgentManager:
        agents, agent_occupancy = build_idm_agents_on_map_rails(self._target_velocity,
                                                                self._min_gap_to_lead_agent,
                                                                self._headway_time,
                                                                self._accel_max,
                                                                self._decel_max,
                                                                self._scenario)

        return IDMAgentManager(agents, agent_occupancy)

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return Detections  # type: ignore

    def get_observation(self) -> Detections:
        """ Inherited, see superclass. """
        detections = self._idm_agent_manager.get_active_agents(self.current_iteration, self._planned_trajectory_samples,
                                                               self._planned_trajectory_sample_interval)
        return detections

    def update_observation(self, iteration: SimulationIteration,
                           next_iteration: SimulationIteration,
                           ego_state: EgoState) -> None:
        """ Inherited, see superclass. """
        self.current_iteration = next_iteration.index
        tspan = next_iteration.time_s - iteration.time_s
        self._idm_agent_manager.propagate_agents(ego_state, tspan, self.current_iteration)
