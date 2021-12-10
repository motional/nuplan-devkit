from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.visualization.abstract_visualization import AbstractVisualization


class VisualizationCallback(AbstractCallback):

    def __init__(self, renderer: AbstractVisualization):
        self._visualization = renderer

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        In initialization start just render scenario
        """
        self._visualization.render_scenario(setup.scenario, True)

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """
        Render sample after a step
        """
        self._visualization.render_ego_state(sample.ego_state)
        self._visualization.render_observations(sample.observation)
        self._visualization.render_trajectory(sample.trajectory.get_sampled_trajectory())
        self._visualization.render(sample.iteration)

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """
        On reached_end just call step_end
        """
        self.on_step_end(setup, planner, history.data[-1])
