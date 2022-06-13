import unittest
from typing import Any, List

import numpy as np

from nuplan.planning.utils.multithreading.worker_utils import chunk_list


class TestChunkSplitter(unittest.TestCase):
    """Unittest class for splitters to chunks"""

    def validate_chunks(self, chunks: List[List[Any]]) -> None:
        """Validate splitter chunks."""
        self.assertTrue(all([len(chunk) > 0 for chunk in chunks]))

    def test_chunk_splitter_more_data_than_number_of_chunks(self) -> None:
        """Test Chunk splitter where"""
        num_variables = 108
        num_chunks = 32
        data = list(range(1, num_variables + 1))
        chunks = chunk_list(data, num_chunks)
        self.validate_chunks(chunks)
        self.assertEqual(len(chunks), num_chunks)
        self.assertLessEqual(max(np.abs(np.diff([len(chunk) for chunk in chunks]))), 1)

    def test_chunk_splitter(self) -> None:
        """Test Chunk splitter where data size is smaller than number of chunks"""
        num_variables = 20
        num_chunks = 32
        data = list(range(1, num_variables + 1))
        chunks = chunk_list(data, num_chunks)
        self.validate_chunks(chunks)
        self.assertLessEqual(max(np.abs(np.diff([len(chunk) for chunk in chunks]))), 1)

    def test_chunk_splitter_same_size(self) -> None:
        """Test Chunk splitter where data and number chunks is the same"""
        num_chunks = 32
        num_variables = num_chunks
        data = list(range(1, num_variables + 1))
        chunks = chunk_list(data, num_chunks)
        self.validate_chunks(chunks)
        self.assertTrue(all([len(chunk) == 1 for chunk in chunks]))


if __name__ == '__main__':
    unittest.main()
