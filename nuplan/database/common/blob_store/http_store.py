import io
import logging
import time
from typing import BinaryIO

import requests

from nuplan.database.common.blob_store.blob_store import BlobStore

logger = logging.getLogger(__name__)


class HttpStore(BlobStore):
    """
    Http blob store. Load blobs from http file server.
    """

    def __init__(self, root_url: str) -> None:
        """
        Initialize HttpStore.
        :param root_url: Root URL containing data.
        """
        assert root_url.startswith('http://') or root_url.startswith('https://'), 'invalid url %s' % root_url
        self._root_url = root_url
        if not self._root_url.endswith('/'):
            self._root_url += '/'

        self._session = requests.Session()

    def _get(self, url: str) -> BinaryIO:
        """
        Get content from URL.
        :param url: URL containing the data.
        :return: Blob binary stream.
        """
        t0 = time.time()
        response = self._session.get(url)
        logger.debug("Done fetching {} in {} seconds.".format(url, time.time() - t0))
        if response.status_code == 200:
            return io.BytesIO(response.content)
        else:
            logger.error("Can not load file from URL: {}".format(url))
            raise RuntimeError(
                'Can not load the file: %s from server, '
                'error: %d, msg: %s.' % (url, response.status_code, response.text)
            )

    def get(self, key: str, check_for_compressed: bool = False) -> BinaryIO:
        """
        Get content from URL.
        :param key: File name.
        :param check_for_compressed: Flag that check for a "<key>+.gzip" file and extracts the <key> file.
        :return: Blob binary stream.
        """
        gzip_path = self._root_url + key + '.gzip'
        if check_for_compressed and self.exists(gzip_path):
            gzip_stream = self._get(gzip_path)
            content: BinaryIO = self._extract_gzip_content(gzip_stream)
        else:
            content = self._get(key)

        return content

    async def get_async(self, key: str) -> BinaryIO:
        """Inherited, see superclass."""
        raise NotImplementedError('Not today.')

    def exists(self, key: str) -> bool:
        """
        Tell if the blob exists.
        :param key: blob path or token.
        :return: True if the blob exists else False.
        """
        url = self._root_url + key
        response = self._session.head(url)

        return response.status_code == 200

    def put(self, key: str, value: BinaryIO) -> None:
        """Inherited, see superclass."""
        raise NotImplementedError("'Put' operation not supported for legacy HttpStore class")

    def save_to_disk(self, key: str, check_for_compressed: bool = False) -> None:
        """Inherited, see superclass."""
        super().save_to_disk(key, check_for_compressed=check_for_compressed)
