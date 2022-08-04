from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuplan.database.nuplan_db_orm.lidar_box import LidarBox


class IterableLidarBox:
    """Helper class to make LidarBox object iterable via for loop."""

    def __init__(self, box: LidarBox, reverse: bool = False) -> None:
        """
        Constructs the iterable object.

        :param box: LidarBox to iterate over.
        :param reverse: if true, this iterator will iterate to past
        """
        self._begin = box
        self._current = box
        self._reverse = reverse
        self._items_dict = box.get_box_items_to_iterate()

    @property
    def box(self) -> LidarBox:
        """
        :return: the current lidar box
        """
        return self._current

    @property
    def reverse(self) -> int:
        """
        :return: whether this class is iterating into past.
        """
        return self._reverse

    def __iter__(self) -> IterableLidarBox:
        """
        Returns the iterable object itself.
        """
        return self

    def __next__(self) -> LidarBox:
        """
        Returns the next LidarBox object.
        """
        box = self._current

        if box:
            # Index 1 meaning we will get the timestamp of the NEXT box
            # Index 0 meaning we will get the timestamp of the PREVIOUS box and thus, we will iterate in reverse
            index = 1 if not self._reverse else 0
            self._current = self._items_dict[box.timestamp][index]

            return box

        raise StopIteration

    def __getitem__(self, item: int) -> LidarBox:
        """
        Returns the LidarBox object at the given index.
        """
        box = self._begin

        for _ in range(item):
            if box:
                box = self._items_dict[box.timestamp][1]

        return box
