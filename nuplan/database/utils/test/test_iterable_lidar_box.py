import unittest
from unittest.mock import Mock

from nuplan.database.utils.iterable_lidar_box import IterableLidarBox


class TestIterableLidarBox(unittest.TestCase):
    """Tests the IterableLidarBox class and it's methods"""

    def test_IterableLidarBox_init(self) -> None:
        """Checks the correctness of the `IterableLidarBox` class' constructor."""
        # Setup
        box = Mock()

        # Call function under test
        iterable = IterableLidarBox(box)

        # Assertions
        box.get_box_items_to_iterate.assert_called_once()
        self.assertEqual(iterable._begin, box)
        self.assertEqual(iterable.box, box)
        self.assertEqual(iterable._current, box)
        self.assertEqual(iterable._reverse, False)
        self.assertEqual(iterable._items_dict, box.get_box_items_to_iterate.return_value)

    def test_IterableLidarBox_iter(self) -> None:
        """Checks the correctness of the `IterableLidarBox` class' `__iter__` method."""
        # Setup
        box = Mock()
        iterable = IterableLidarBox(box)

        # Call function under test
        iter_val = iter(iterable)

        # Assertions
        self.assertEqual(iterable, iter_val)

    def test_IterableLidarBox_next_not_end(self) -> None:
        """
        Checks the correctness of the `IterableLidarBox` class' `__next__` method.
         When the current box is not the end box, the current box should be returned
         and the `.next` box should become the current one.
        """
        # Setup
        box = Mock()
        box.timestamp = 1
        box.get_box_items_to_iterate.return_value = {1: (Mock(), Mock())}

        # Call function under test
        iterable = IterableLidarBox(box)
        result = next(iterable)

        # Assertions
        self.assertEqual(result, box)
        self.assertEqual(iterable._current, box.get_box_items_to_iterate.return_value[1][1])

    def test_IterableLidarBox_getitem(self) -> None:
        """
        Checks the correctness of the `IterableLidarBox` class' `__getitem__` method.
        Should return the box at the given index.
        """
        # Setup
        box = Mock()
        box.timestamp = 1
        box.get_box_items_to_iterate.return_value = {1: (Mock(), Mock())}
        iterable = IterableLidarBox(box)

        # Call function under test
        box_0 = iterable[0]

        # Assertions
        self.assertEqual(box_0, box)


if __name__ == '__main__':
    unittest.main()
