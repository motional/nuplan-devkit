import asyncio
import unittest

from bokeh.io import output_notebook
from bokeh.io.state import curstate
from tutorials.utils.tutorial_utils import (
    get_default_scenario_from_token,
    get_scenario_type_token_map,
    start_event_loop_if_needed,
    visualize_nuplan_scenarios,
    visualize_scenario,
)

from nuplan.database.tests.test_utils_nuplan_db import (
    NUPLAN_DATA_ROOT,
    NUPLAN_DB_FILES,
    NUPLAN_MAP_VERSION,
    NUPLAN_MAPS_ROOT,
)


class TestTutorialUtils(unittest.TestCase):
    """Unit tests for tutorial_utils.py."""

    def test_scenario_visualization_utils(self) -> None:
        """Test if scenario visualization utils work as expected."""
        visualize_nuplan_scenarios(
            data_root=NUPLAN_DATA_ROOT,
            db_files=NUPLAN_DB_FILES,
            map_root=NUPLAN_MAPS_ROOT,
            map_version=NUPLAN_MAP_VERSION,
        )

    def test_scenario_rendering(self) -> None:
        """Test if scenario rendering works."""
        bokeh_port = 8999

        output_notebook()
        scenario_type_token_map = get_scenario_type_token_map(NUPLAN_DB_FILES)
        available_keys = list(scenario_type_token_map.keys())
        log_db, token = scenario_type_token_map[available_keys[0]][0]
        scenario = get_default_scenario_from_token(
            NUPLAN_DATA_ROOT, log_db, token, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION
        )

        # Visualize twice to make sure bokeh port collision doesn't raise
        for _ in range(2):
            visualize_scenario(scenario, bokeh_port=bokeh_port)

        # Check that any servers bokeh started are assigned to `bokeh_port`
        for server in curstate().uuid_to_server.values():
            self.assertEqual(bokeh_port, server.port)

    def test_start_event_loop_if_needed(self) -> None:
        """Tests if start_event_loop_if_needed works."""

        async def test_fn() -> int:
            """Minimal async function"""
            return 1

        # Event loop is auto-started
        start_event_loop_if_needed()
        _ = asyncio.get_event_loop()

        # Gets running event loop
        start_event_loop_if_needed()
        _ = asyncio.get_event_loop()

        # Stops event loop
        asyncio.run(test_fn())

        with self.assertRaises(RuntimeError):
            _ = asyncio.get_event_loop()

        start_event_loop_if_needed()
        _ = asyncio.get_event_loop()


if __name__ == '__main__':
    unittest.main()
