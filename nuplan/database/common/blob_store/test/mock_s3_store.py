from typing import Any, BinaryIO, Dict, List

from nuplan.database.common.blob_store.blob_store import BlobStore


class MockS3Store(BlobStore):
    """
    Mock S3 blob store for testing `S3Store` class.
    """

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> None:
        """Initialize the mock object."""
        self.store: Dict[str, BinaryIO] = dict()

    def get(self, key: str, check_for_compressed: bool = False) -> BinaryIO:
        """Inherited, see superclass."""
        return self.store[key]

    def exists(self, key: str) -> bool:
        """Inherited, see superclass."""
        return key in self.store

    def put(self, key: str, value: BinaryIO, ignore_if_client_error: bool = False) -> None:
        """Inherited, see superclass."""
        self.store[key] = value

    def save_to_disk(self, key: str, check_for_compressed: bool = False) -> None:
        """Inherited, see superclass."""
        raise NotImplementedError
