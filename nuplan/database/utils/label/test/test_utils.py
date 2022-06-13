import unittest
from collections import OrderedDict

from nuplan.database.utils.label.label import Label
from nuplan.database.utils.label.utils import parse_labelmap_dataclass


class TestParseLabelmap(unittest.TestCase):
    """Test Parsing LabMap."""

    def setUp(self) -> None:
        """Setup function."""
        self.label1 = Label('label1', (1, 1, 1, 1))
        self.label2 = Label('label2', (2, 2, 2, 2))

    def test_empty(self) -> None:
        """Tests empty label map case."""
        id2name, id2color = parse_labelmap_dataclass({})

        # Check it returns ordered dicts.
        self.assertIsInstance(id2name, OrderedDict)
        self.assertIsInstance(id2color, OrderedDict)

        # Both dict should be empty.
        self.assertEqual(len(id2name), 0)
        self.assertEqual(len(id2color), 0)

    def test_one(self) -> None:
        """Tests one label case."""
        num = 1
        mapping = {num: self.label1}
        id2name, id2color = parse_labelmap_dataclass(mapping)
        self.assertEqual(len(id2name), len(mapping))
        self.assertEqual(id2name[num], self.label1.name)
        self.assertEqual(len(id2color), len(mapping))
        self.assertEqual(id2color[num], self.label1.color)

    def test_multiple(self) -> None:
        """Tests multiple labels case."""
        num1, num2 = 1, 2
        mapping = {num1: self.label1, num2: self.label2}
        id2name, id2color = parse_labelmap_dataclass(mapping)
        self.assertEqual(len(id2name), len(mapping))
        self.assertEqual(len(id2color), len(mapping))

        # Check correct name and colors are extracted.
        self.assertEqual(id2name[num1], self.label1.name)
        self.assertEqual(id2name[num2], self.label2.name)
        self.assertEqual(id2color[num1], self.label1.color)
        self.assertEqual(id2color[num2], self.label2.color)

        # Make sure the dicts are sorted by label id
        self.assertEqual(list(id2name.keys())[0], min(num1, num2))
        self.assertEqual(list(id2name.keys())[1], max(num1, num2))
        self.assertEqual(list(id2color.keys())[0], min(num1, num2))
        self.assertEqual(list(id2color.keys())[1], max(num1, num2))


if __name__ == '__main__':
    unittest.main()
