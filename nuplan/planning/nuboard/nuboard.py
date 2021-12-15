import logging
import os
import signal
from pathlib import Path
from typing import Any, List, Optional

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document.document import Document
from bokeh.layouts import layout
from bokeh.models import Div, Tabs
from bokeh.server.server import Server
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.nuboard.tabs.configuration_tab import ConfigurationTab
from nuplan.planning.nuboard.tabs.histogram_tab import HistogramTab
from nuplan.planning.nuboard.tabs.overview_tab import OverviewTab
from nuplan.planning.nuboard.tabs.scenario_tab import ScenarioTab
from nuplan.planning.nuboard.utils.utils import check_nuboard_file_paths, read_nuboard_file_paths
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.training.callbacks.profile_callback import ProfileCallback
from tornado.ioloop import IOLoop
from tornado.web import StaticFileHandler

logger = logging.getLogger(__name__)


class NuBoard:

    def __init__(self,
                 nuboard_paths: List[str],
                 scenario_builder: AbstractScenarioBuilder,
                 metric_categories: List[str],
                 vehicle_parameters: VehicleParameters,
                 port_number: int = 5006,
                 profiler_path: Optional[Path] = None,
                 resource_prefix: Optional[str] = None):
        """
        Nuboard main class.
        :param nuboard_paths: A list of paths to nuboard files.
        :param scenario_builder: Scenario builder instance.
        :param metric_categories A list of metric categories.
        :param vehicle_parameters: vehicle parameters.
        :param port_number: Bokeh port number.
        :param profiler_path: Path to save the profiler.
        :param resource_prefix: Prefix to the resource path in HTML.
        """

        self._profiler_path = profiler_path
        self._nuboard_paths = check_nuboard_file_paths(nuboard_paths)
        self._scenario_builder = scenario_builder
        self._metric_categories = metric_categories
        self._port_number = port_number
        self._vehicle_parameters = vehicle_parameters
        self._doc = None
        self._resource_prefix = resource_prefix if resource_prefix else ''
        self._resource_path = os.path.normpath(os.path.dirname(__file__) + '/resource')
        self._profiler_file_name = "nuboard"
        self._profiler: Optional[ProfileCallback] = None

    def stop_handler(self, sig: Any, frame: Any) -> None:
        """ Helper to handle stop signals. """

        logger.info("Stopping the Bokeh application.")
        if self._profiler:
            self._profiler.save_profiler(self._profiler_file_name)
        IOLoop.current().stop()

    def run(self) -> None:
        logger.info(f'Opening Bokeh application on http://localhost:{self._port_number}/')

        io_loop = IOLoop.current()

        # Add stopping signal
        # Todo: Extract profiler to a more general interface
        if self._profiler is not None:
            signal.signal(signal.SIGTERM, self.stop_handler)
            signal.signal(signal.SIGINT, self.stop_handler)

            # Add profiler
            self._profiler = ProfileCallback(output_dir=self._profiler_path)
            self._profiler.start_profiler(self._profiler_file_name)

        bokeh_app = Application(FunctionHandler(self.main_page))
        server = Server({'/': bokeh_app}, io_loop=io_loop, port=self._port_number,
                        extra_patterns=[(r'/resource/(.*)', StaticFileHandler, {'path': self._resource_path})])
        server.start()

        io_loop.add_callback(server.show, "/")
        io_loop.start()

    def main_page(self, doc: Document) -> None:
        """
        Main bokeh page.
        :param doc: HTML document.
        """

        self._doc = doc
        doc.title = 'nuPlan Dashboard'
        header = Div(
            text=f"<link rel='stylesheet' type='text/css' href='{self._resource_prefix}resource/style.css'>")
        self._doc.add_root(header)  # type: ignore

        div_title = Div(text=f"""<img src='{self._resource_prefix}resource/motional_logo.png'
                             style='vertical-align: middle; width: 200px;height:200px'>
                             <h1 style='display: inline; vertical-align: middle; color: #5C48F6; font-size: 50px;
                             margin-left: 50px;'>
                             nuBoard</h1>""", width=800)

        nuboard_files = read_nuboard_file_paths(file_paths=self._nuboard_paths)

        self._doc.add_root(layout(div_title))  # type: ignore
        overview_tab = OverviewTab(file_paths=nuboard_files, doc=self._doc, metric_categories=self._metric_categories)
        histogram_tab = HistogramTab(file_paths=nuboard_files, doc=self._doc)
        scenario_tab = ScenarioTab(file_paths=nuboard_files, scenario_builder=self._scenario_builder, doc=self._doc,
                                   vehicle_parameters=self._vehicle_parameters)
        configuration_tab = ConfigurationTab(file_paths=nuboard_files, doc=self._doc,
                                             tabs=[overview_tab, histogram_tab, scenario_tab])

        tabs = [configuration_tab.panel, overview_tab.panel, histogram_tab.panel, scenario_tab.panel]
        panel_tabs = Tabs(tabs=tabs)

        self._doc.add_root(panel_tabs)  # type: ignore
