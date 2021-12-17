from typing import Any, List, Optional

import numpy as np
from psutil import cpu_count


def chunk_list(input_list: List[Any], num_chunks: Optional[int] = None) -> List[List[Any]]:
    """
    Chunks a list to equal sized lists. The size of the last list might be truncated.
    :param input_list: List to be chunked.
    :param num_chunks: Number of chunks, equals to the number of cores if set to None.
    :return: List of equal sized lists.
    """
    num_chunks = num_chunks if num_chunks else cpu_count(logical=True)
    chunks = np.array_split(input_list, num_chunks)  # type: ignore
    return [chunk.tolist() for chunk in chunks if len(chunk) != 0]
