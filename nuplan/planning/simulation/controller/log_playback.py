from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory


class LogPlaybackController(AbstractEgoController):
    """
    Replay the GT position from samples
    """

    def __init__(self, scenario: AbstractScenario):
        assert isinstance(scenario, AbstractScenario)  # This class is only defined for open loop

        self.scenario = scenario
        self.current_iteration = 0

    def get_state(self) -> EgoState:
        """ Inherited, see superclass. """
        return self.scenario.get_ego_state_at_iteration(self.current_iteration)

    def update_state(self,
                     current_iteration: SimulationIteration,
                     next_iteration: SimulationIteration,
                     ego_state: EgoState,
                     trajectory: AbstractTrajectory) -> None:
        """ Inherited, see superclass. """
        self.current_iteration = next_iteration.index
        # TODO: We need to update ego_state here.
