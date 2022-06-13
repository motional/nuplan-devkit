import json
import unittest

from nuplan.database.utils.label.label import Label


class TestLabel(unittest.TestCase):
    """Test Label Serialization."""

    def test_serialize(self) -> None:
        """Tests a serialized label are still the same after serializing."""
        label = Label('my_name', (1, 3, 4, 1))
        self.assertEqual(label, Label.deserialize(json.loads(json.dumps(label.serialize()))))


if __name__ == '__main__':
    unittest.main()
