from __future__ import annotations

import io
import os
from pathlib import Path
from typing import BinaryIO, Tuple, Type

from nuplan.database.common.blob_store.blob_store import BlobStore, BlobStoreKeyNotFound


class LocalStore(BlobStore):
    """
    Local blob store. Load blobs from local file system.
    """

    def __init__(self, root_dir: str) -> None:
        """
        Initialize LocalStore.
        :param root_dir: Root directory containing the data.
        """
        self._root_dir = root_dir
        assert os.path.isdir(self._root_dir), '%s does not exist!' % self._root_dir
        assert os.access(self._root_dir, os.R_OK | os.X_OK), 'can not read from %s' % self._root_dir

    def __reduce__(self) -> Tuple[Type[LocalStore], Tuple[str]]:
        """
        :return: Tuple of class and its constructor parameters, this is used to pickle the class.
        """
        return self.__class__, (self._root_dir,)

    def get(self, key: str, check_for_compressed: bool = False) -> BinaryIO:
        """
        Get blob content.
        :param key: Blob path or token.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        :raises: BlobStoreKeyNotFound is `key` is not present in backing store.
        :return: A file-like object, use read() to get raw bytes.
        """
        path = os.path.join(self._root_dir, key)
        try:
            with open(path, 'rb') as fp:
                return io.BytesIO(fp.read())
        except FileNotFoundError as e:
            raise BlobStoreKeyNotFound(e)

    def save_to_disk(self, key: str, check_for_compressed: bool = False) -> None:
        """
        Save content to disk.
        :param key:. Blob path or token.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        """
        # With LocalStore the data is already saved to disk, so this function
        # doesn't need to do anything.
        pass

    async def get_async(self, key: str) -> BinaryIO:
        """Inherited, see superclass."""
        raise NotImplementedError('Not today.')

    def exists(self, key: str) -> bool:
        """
        Tell if the blob exists.
        :param key: blob path or token.
        :return: True if the blob exists else False.
        """
        path = os.path.join(self._root_dir, key)
        return os.path.isfile(path)

    def put(self, key: str, value: BinaryIO) -> None:
        """
        Writes content.
        :param key: Blob path or token.
        :param value: Data to save.
        """
        if not os.access(self._root_dir, os.W_OK):
            raise RuntimeError(f"No write access to {self._root_dir}")

        path = Path(self._root_dir) / key
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            f.write(value.read())
