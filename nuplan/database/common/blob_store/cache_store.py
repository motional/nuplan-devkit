from __future__ import annotations

import os
from typing import BinaryIO, Set, Tuple, Type

from nuplan.database.common.blob_store.blob_store import BlobStore
from nuplan.database.common.blob_store.local_store import LocalStore


class CacheStore(BlobStore):
    """
    Cache store, it combines a remote blob store and local store. The idea is to load blob
    from a remote store and cache it in local store so the next time we can load it from
    local.
    """

    def __init__(self, cache_dir: str, remote: BlobStore) -> None:
        """
        Initialize CacheStore.
        :param cache_dir: Path where to cache.
        :param remote: BlobStore instance.
        """
        os.makedirs(cache_dir, exist_ok=True)

        self._local = LocalStore(cache_dir)
        self._cache_dir = cache_dir
        self._remote = remote
        self._on_disk: Set[str] = set()

    def __reduce__(self) -> Tuple[Type[CacheStore], Tuple[str, BlobStore]]:
        """
        :return: tuple of class and its constructor parameters, this is used to pickle the class.
        """
        return self.__class__, (self._cache_dir, self._remote)

    def get(self, key: str, check_for_compressed: bool = False) -> BinaryIO:
        """
        Get blob content if its present. Else download and then return.
        :param key: Blob path or token.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        :return: A file-like object, use read() to get raw bytes.
        """
        if self.exists(key):
            content: BinaryIO = self._local.get(key)
        else:
            content = self._remote.get(key, check_for_compressed)
            key_split = key.split('/')
            self.save(key_split[-1], content)
            content.seek(0)

        return content

    def save_to_disk(self, key: str, check_for_compressed: bool = False) -> None:
        """
        Save content to disk.
        :param key: Blob path or token.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        """
        if not self.exists(key):
            content = self._remote.get(key, check_for_compressed)
            self.save(key, content)

    async def get_async(self, key: str) -> BinaryIO:
        """Inherited, see superclass."""
        raise NotImplementedError('Not today.')

    def exists(self, key: str) -> bool:
        """
        Check if the blob exists.
        :param key: blob path or token.
        :return: True if the blob exists else False.
        """
        if key in self._on_disk:
            return True

        if self._local.exists(key):
            self._on_disk.add(key)
            return True

        return False

    def put(self, key: str, value: BinaryIO) -> None:
        """
        Write content.
        :param key: Blob path or token.
        :param value: Data to save.
        """
        self._remote.put(key, value)
        value.seek(0)
        self._local.put(key, value)
        self._on_disk.add(key)

    def save(self, key: str, content: BinaryIO) -> None:
        """
        Save to disk.
        :param key: Blob path or token.
        :param content: Data to save.
        """
        assert os.access(self._cache_dir, os.W_OK), 'Can not write to %s' % self._cache_dir

        path = os.path.join(self._cache_dir, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as fp:
            fp.write(content.read())
