import unittest

from bokeh.io import output_notebook
from tutorials.utils.tutorial_utils import (
    get_default_scenario_from_token,
    get_scenario_type_token_map,
    visualize_nuplan_scenarios,
    visualize_scenario,
)

from nuplan.database.tests.nuplan_db_test_utils import (
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
        output_notebook()
        scenario_type_token_map = get_scenario_type_token_map(NUPLAN_DB_FILES)
        available_keys = list(scenario_type_token_map.keys())
        log_db, token = scenario_type_token_map[available_keys[0]][0]
        scenario = get_default_scenario_from_token(
            NUPLAN_DATA_ROOT, log_db, token, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION
        )
        visualize_scenario(scenario)


if __name__ == '__main__':
    unittest.main()
