import os
import unittest
from pathlib import Path

from bokeh.document.document import Document
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.nuboard.base.data_class import SimulationScenarioKey
from nuplan.planning.nuboard.base.simulation_tile import SimulationTile
from nuplan.planning.scenario_builder.test.mock_abstract_scenario_builder import MockAbstractScenarioBuilder


class TestSimulationTile(unittest.TestCase):
    """ Test simulation_tile functionality. """

    def setUp(self) -> None:
        """ Set up simulation tile. """

        vehicle_parameters = get_pacifica_parameters()
        json_main_path = os.path.dirname(os.path.realpath(__file__))
        self.scenario_keys = [
            SimulationScenarioKey(planner_name='SimplePlanner',
                                  scenario_type='common',
                                  scenario_name='test',
                                  files=[
                                      Path(json_main_path) / "json/test_simulation_tile.json"
                                  ]
                                  )
        ]
        doc = Document()
        scenario_builder = MockAbstractScenarioBuilder()
        self.simulation_tile = SimulationTile(
            doc=doc,
            scenario_builder=scenario_builder,
            vehicle_parameters=vehicle_parameters,
            radius=80
        )

    def test_simulation_tile_layout(self) -> None:
        """ Test layout design. """

        layout = self.simulation_tile.render_simulation_tiles(selected_scenario_keys=self.scenario_keys)
        self.assertEqual(len(layout), 1)
