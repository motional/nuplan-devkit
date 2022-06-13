import pickle
import unittest

from nuplan.database.nuplan_db.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.tests.nuplan_db_test_utils import get_test_nuplan_db_wrapper


class TestNuPlanDBWrapper(unittest.TestCase):
    """Test NuPlanDB wrapper which supports loading/accessing multiple log databases."""

    def setUp(self) -> None:
        """Set up test case."""
        self.db_wrapper = get_test_nuplan_db_wrapper()

    def test_serialization(self) -> None:
        """Test whether the wrapper object can be serialized/deserialized correctly."""
        serialized_binary = pickle.dumps(self.db_wrapper)
        re_db_wrapper: NuPlanDBWrapper = pickle.loads(serialized_binary)

        self.assertEqual(self.db_wrapper.data_root, re_db_wrapper.data_root)

    def test_maps_db(self) -> None:
        """Test that maps DB has been loaded."""
        self.db_wrapper.maps_db.load_vector_layer('us-nv-las-vegas-strip', 'lane_connectors')


if __name__ == '__main__':
    unittest.main()
