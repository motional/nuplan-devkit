import abc
import gzip
import io
from typing import Any, BinaryIO


class BlobStoreKeyNotFound(ValueError):
    """Error raised when blob store key is not found."""

    def __init__(self, *args: Any) -> None:
        """
        :param args: Arguments.
        """
        super().__init__(*args)


class BlobStore(abc.ABC):
    """
    BlobStore interface, the idea is to abstract the way we load blob content.
    """

    @abc.abstractmethod
    def get(self, key: str, check_for_compressed: bool = False) -> BinaryIO:
        """
        Get blob content.
        :param key: Blob path or token.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        :raises: BlobStoreKeyNotFound is `key` is not present in backing store.
        :return: A file-like object, use read() to get raw bytes.
        """
        pass

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """
        Tell if the blob exists.
        :param key: blob path or token.
        :return: True if the blob exists else False.
        """
        pass

    @abc.abstractmethod
    def put(self, key: str, value: BinaryIO) -> None:
        """
        Writes content to the blobstore.
        :param key: Blob path or token.
        :param value: Data to save.
        """
        pass

    @abc.abstractmethod
    def save_to_disk(self, key: str, check_for_compressed: bool = False) -> None:
        """
        Save the data to disk.
        :param key: Blob path or token.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        """
        pass

    def _extract_gzip_content(self, gzip_stream: BinaryIO) -> BinaryIO:
        """
        Decompress data.
        :param gzip_stream: Data to decompress.
        :return: Extracted binary data.
        """
        decompressed = gzip.decompress(gzip_stream.read())
        return io.BytesIO(decompressed)
