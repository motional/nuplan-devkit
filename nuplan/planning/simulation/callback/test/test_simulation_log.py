import unittest
from pathlib import Path

from nuplan.planning.simulation.simulation_log import SimulationLog


class TestSimulationLog(unittest.TestCase):
    """Tests metrics callback."""

    def test_simulation_log_type(self) -> None:
        """Checks for the expected behavior of the simulation_log_type function."""
        # Nominal cases: msgpack
        for path in (
            Path("/foo.msgpack.xz"),
            Path("/foo/bar/baz/1.2.msgpack.xz"),
            Path("/foo/bar/baz.1.2.pickle.msgpack.xz"),  # will be treated as msgpack
            Path(
                # real-world test case:
                "/data/exp/username/mmdbqsb_test/simulation_simple_experiment/open_loop_boxes/2022.12.14.13.48.22/simulation_log/SimplePlanner/unknown/2021.09.01.07.19.19_g1p-veh-2051/2021.09.01.07.19.19_g1p-veh-2051_0000074/2021.09.01.07.19.msgpack.xz"
            ),
        ):
            self.assertEqual(SimulationLog.simulation_log_type(path), "msgpack")

        # Nominal cases: msgpack
        for path in (
            Path("/foo.pkl.xz"),
            Path("/foo/bar/baz.1.2.pkl.xz"),
            Path("/foo/bar/baz.1.2.msgpack.pkl.xz"),  # will be treated as msgpack
            Path(
                # real-world test case:
                "/data/exp/username/mmdbqsb_test/simulation_simple_experiment/open_loop_boxes/2022.12.14.13.48.22/simulation_log/SimplePlanner/unknown/2021.09.01.07.19.19_g1p-veh-2051/2021.09.01.07.19.19_g1p-veh-2051_0000074/2021.09.01.07.19.pkl.xz"
            ),
        ):
            self.assertEqual(SimulationLog.simulation_log_type(path), "pickle")

        # Failing cases
        for path in (
            Path("/foo"),
            Path("/foo.pkl"),
            Path("/foo.msgpack"),
            Path("/foo/bar/baz.1.2.pkl"),
            Path("/foo/bar/baz.1.2.msgpack"),
            Path("/foo/bar/baz.1.2.pkl.msgpack"),
            Path("/foo/bar/baz.1.2.xz"),
            Path("/foo/bar/baz.1.2.json.xz"),
        ):
            with self.assertRaises(ValueError):
                SimulationLog.simulation_log_type(path)


if __name__ == '__main__':
    unittest.main()
