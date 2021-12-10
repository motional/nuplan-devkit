from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.trajectory import AbstractTrajectory


class AbstractCallback:

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Called when initialization of simulation starts.
        :param setup: simulation setup
        :param planner: planner before initialization
        """
        pass

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Called when initialization of simulation ends.
        :param setup: simulation setup
        :param planner: planner after initialization
        """
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Called when simulation step starts.
        :param setup: simulation setup
        :param planner: planner at start of a step
        """
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """
        Called when simulation step ends.
        :param setup: simulation setup
        :param planner: planner at end of a step
        :param sample: result of a step
        """
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Called when planner starts to compute trajectory.
        :param setup: simulation setup
        :param planner: planner before planner.compute_trajectory() is called
        """
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """
        Called when planner ends to compute trajectory.
        :param setup: simulation setup
        :param planner: planner after planner.compute_trajectory() is called
        :param trajectory: trajectory resulting from planner
        """
        pass

    def on_simulation_manager_end(self,
                                  setup: SimulationSetup,
                                  planner: AbstractPlanner,
                                  history: SimulationHistory) -> None:
        """
        Called when simulation manager reaches end of iterations.
        :param setup: simulation setup
        :param planner: when simulation manager terminates simulation
        :param history: when simulation manager terminates simulation
        """
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """
        Called when simulation starts.
        :param setup: simulation setup
        """
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """
        Called when simulation ends.
        :param setup: simulation setup
        :param planner: planner when simulation ends
        :param history: resulting from simulation
        """
        pass
