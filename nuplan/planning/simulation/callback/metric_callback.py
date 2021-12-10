import logging

from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup

logger = logging.getLogger(__name__)


class MetricCallBack(AbstractCallback):

    def __init__(self, metric_engine: MetricsEngine, scenario_name: str):
        """
        Build A metric callback.
        :param metric_engine: Metric Engine.
        :param scenario_name: Name of the scenario on which the callback runs
        """

        self._metric_engine = metric_engine
        self._scenario_name = scenario_name

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """ Inherited, see superclass. """
        logger.info("Starting metrics computation...")
        metric_files = self._metric_engine.compute(history, scenario_name=self._scenario_name,
                                                   planner_name=planner.name())
        logger.info("Finished metrics computation!")
        self._metric_engine.save_metric_files(metric_files)
        logger.info("Saved metrics!")
