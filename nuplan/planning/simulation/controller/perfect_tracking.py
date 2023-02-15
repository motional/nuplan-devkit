from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class PerfectTrackingController(AbstractEgoController):
    """
    Assume tracking controller is absolutely perfect, and just follow a trajectory.
    """

    def __init__(self, scenario: AbstractScenario):
        """
        Constructor of PerfectTrackingController.
        :param scenario: scenario to run through.
        """
        self.scenario = scenario
        self.current_state = None  # set to None to allow lazily loading of the scenario's initial ego state

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_state = None

    def get_state(self) -> EgoState:
        """Inherited, see superclass."""
        if self.current_state is None:
            self.current_state = self.scenario.initial_ego_state

        return self.current_state

    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> None:
        """Inherited, see superclass."""
        self.current_state = trajectory.get_state_at_time(next_iteration.time_point)
        assert self.current_state is not None, 'Current state of controller cannot be None'

        # Raise in case the desired velocity is extremely high
        very_large_velocity_threshold = 50
        if self.current_state.dynamic_car_state.speed >= very_large_velocity_threshold:
            raise RuntimeError(f"Velocity is too high: {self.current_state.dynamic_car_state.speed}!")
