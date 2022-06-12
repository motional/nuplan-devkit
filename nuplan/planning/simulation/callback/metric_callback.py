import logging

from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup

logger = logging.getLogger(__name__)


class MetricCallback(AbstractCallback):
    """Callback for computing metrics at the end of the simulation."""

    def __init__(self, metric_engine: MetricsEngine):
        """
        Build A metric callback.
        :param metric_engine: Metric Engine.
        """
        self._metric_engine = metric_engine

    def run_metric_engine(self, scenario: AbstractScenario, planner_name: str, history: SimulationHistory) -> None:
        """
        Run the metric engine.
        """
        logger.info("Starting metrics computation...")
        metric_files = self._metric_engine.compute(history, scenario=scenario, planner_name=planner_name)
        logger.info("Finished metrics computation!")
        logger.info("Saving metric statistics!")
        self._metric_engine.write_to_files(metric_files)
        logger.info("Saved metrics!")

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Inherited, see superclass."""
        self.run_metric_engine(history=history, scenario=setup.scenario, planner_name=planner.name())
