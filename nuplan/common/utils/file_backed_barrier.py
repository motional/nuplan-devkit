import logging
import os
import ssl
import time
from contextlib import closing
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import urllib3
from botocore.exceptions import BotoCoreError, NoCredentialsError

# retry type stubs only support python 3.8 or earlier
from retry import retry  # type: ignore

from nuplan.common.utils.s3_utils import get_s3_client

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    urllib3.exceptions.ProtocolError,
    urllib3.exceptions.SSLError,
    ssl.SSLError,
    BotoCoreError,
    NoCredentialsError,
)

# Multiplier for how long to sleep after all processes have completed, before cleaning up files
# sleep time = multiplier * poll_interval_s
SLEEP_MULTIPLIER_BEFORE_CLEANUP = 20


class FileBackedBarrier:
    """
    A file-based synchronization barrier.
    This class can be used to synchronize activies across multiple machines.
    """

    def __init__(self, barrier_directory: Path) -> None:
        """
        Initializes a FileBackedBarrier.
        :param barrier_directory: The path that the barrier files will use for synchronization.
          This can be a local or S3 path.
        """
        self._barrier_directory = barrier_directory
        self._is_s3 = str(barrier_directory).startswith("s3:")
        self._activity_file_content = "x"  # an arbitrary string.

    def wait_barrier(
        self,
        activity_id: str,
        expected_activity_ids: Set[str],
        timeout_s: Optional[float] = None,
        poll_interval_s: float = 1,
    ) -> None:
        """
        Registers that `activity_id` has completed.
        Waits until all activities in `expected_activity_ids` have completed.
        If timeout_s has been provided, the operation will raise a TimeoutError after
          the supplied number of seconds has passed.

        :param activity_id: The activity ID that will be registered as completed.
        :param expected_activity_ids: The list of activity IDs that are expected to be completed.
          The function will block until these are done.
        :param timeout_s: If provided, the timeout for the wait operation.
          If the operation does not complete within this amount of time, then a TimeoutError will be raised.
        :param poll_interval_s: The elapsed time before polling for new files.
        """
        logger.info("Writing completion of activity id %s to directory %s...", activity_id, self._barrier_directory)
        self._register_activity_id_complete(activity_id)

        logger.info("Waiting for all processes to finish processing")
        self._wait(expected_activity_ids, timeout_s, poll_interval_s)

        logger.info(
            f"Sleeping for {poll_interval_s * SLEEP_MULTIPLIER_BEFORE_CLEANUP} seconds so that the other processes catch up before moving on"
        )
        time.sleep(poll_interval_s * SLEEP_MULTIPLIER_BEFORE_CLEANUP)

        logger.info("All Processes Synced, clearing activity file")
        self._remove_activity_after_processing(activity_id)

        logger.info("Waiting for all processes to clean up barrier files")
        self._wait(set(), timeout_s, poll_interval_s)

    def _wait(
        self, expected_activity_ids: Set[str], timeout_s: Optional[float] = None, poll_interval_s: float = 1
    ) -> None:
        start_wait_time = time.time()
        logger.info("Beginning barrier wait at time %f", start_wait_time)
        while True:
            next_wait_time = time.time() + poll_interval_s
            logger.debug("The next wait time is %f. Getting completed activity ids...", next_wait_time)
            completed_activity_ids = self._get_completed_activity_ids()

            logger.debug("There are %d completed activities.", len(completed_activity_ids))

            if expected_activity_ids == completed_activity_ids:
                logger.debug("All activities completed! Ending wait.")
                return

            total_wait_time = time.time() - start_wait_time
            logger.debug("All tasks not finished. Total elapsed wait time is %f.", total_wait_time)
            if timeout_s is not None and total_wait_time > timeout_s:
                raise TimeoutError(
                    f"Waited {total_wait_time} sec for barrier {self._barrier_directory}, which is longer than configured timeout of {timeout_s}."
                )

            sleep_time = max(0.0, next_wait_time - time.time())
            logger.debug("Sleeping for %f seconds.", sleep_time)
            time.sleep(sleep_time)

    def _register_activity_id_complete(self, activity_id: str) -> None:
        """
        Registers an activity_id as completed by creating a file in the configured directory.
        :param activity_id: The activity ID to register as completed.
        """
        activity_id_file_path = self._barrier_directory / activity_id
        if self._is_s3:
            s3_bucket, s3_key = self._split_s3_path(activity_id_file_path)
            self._create_activity_file_in_s3(s3_key, s3_bucket)
        else:
            activity_id_file_path.parent.mkdir(exist_ok=True, parents=True)
            with open(activity_id_file_path, "w") as f:
                f.write(self._activity_file_content)

    def _get_completed_activity_ids(self) -> Set[str]:
        """
        Gets the activity IDs from the filesystem that have been marked as completed.
        :return: The completed file system activity ids.
        """
        if self._is_s3:
            s3_bucket, s3_key = self._split_s3_path(self._barrier_directory)
            files = [Path(p) for p in self._list_files_in_s3_directory(s3_key, s3_bucket)]
        else:
            files = [x for x in self._barrier_directory.iterdir() if x.is_file()]

        unique_activity_ids = {f.stem for f in files}

        return unique_activity_ids

    def _remove_activity_after_processing(self, activity_id: str) -> None:
        """
        Removes the activity file so that we can reuse the same directory in future calls to sync
        """
        activity_id_file_path = self._barrier_directory / activity_id
        if self._is_s3:
            s3_bucket, s3_key = self._split_s3_path(activity_id_file_path)
            self._remove_activity_file_from_s3(s3_key, s3_bucket)
        else:
            activity_id_file_path.unlink()

    @retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
    def _create_activity_file_in_s3(self, s3_key: Path, s3_bucket: str) -> None:
        """
        Creates an activity file in S3
        :param s3_key: The S3 path for the file, without the bucket.
        :param s3_bucket: The name of the bucket to write to.
        """
        with closing(get_s3_client()) as s3_client:
            logger.info(f"Creating activity file at {s3_key} in bucket {s3_bucket}...")
            s3_client.put_object(Body=self._activity_file_content.encode("utf-8"), Bucket=s3_bucket, Key=str(s3_key))

    @retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
    def _remove_activity_file_from_s3(self, s3_key: Path, s3_bucket: str) -> None:
        """
        Creates an activity file in S3
        :param s3_key: The S3 path for the file, without the bucket.
        :param s3_bucket: The name of the bucket to write to.
        """
        with closing(get_s3_client()) as s3_client:
            logger.info(f"Removing activity file at {s3_key} in bucket {s3_bucket}...")
            s3_client.delete_object(Bucket=s3_bucket, Key=str(s3_key))

    @retry(RETRYABLE_EXCEPTIONS, backoff=1, tries=3, delay=0.5)
    def _list_files_in_s3_directory(self, s3_key: Path, s3_bucket: str) -> List[Path]:
        """
        Lists the files available in a particular S3 directory.
        :param s3_key: The path to list, without the bucket.
        :param s3_bucket: The bucket to list.
        :return: The files in the folder.
        """
        with closing(get_s3_client()) as s3_client:
            # S3 directories need to have closing slash.
            # Otherwise, just the directory name is returned.
            key = str(s3_key)
            if not key.endswith("/"):
                key += "/"

            objects = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=key)

            if "Contents" in objects:
                return [Path(k["Key"]) for k in objects["Contents"]]
            return []

    def _split_s3_path(self, s3_path: Path) -> Tuple[str, Path]:
        """
        Splits a S3 path into a (bucket, path) set of identifiers.
        :param s3_path: The full S3 path.
        :return: A tuple of (bucket, path).
        """
        # Expect path of the form:
        # s3://ml-caches/folder/folder2/folder3/file.txt
        #
        # Would result in:
        #  bucket = "ml-caches"
        #  path = "folder/folder2/folder3/file.txt"
        chunks = [v.strip() for v in str(s3_path).split("/") if len(v.strip()) > 0]

        bucket = chunks[1]
        path = Path("/".join(chunks[2:]))

        return (bucket, path)


def distributed_sync(path: Union[Path, str], timeout_seconds: int = 7200, poll_interval: float = 0.5) -> None:
    """
    Use a FileBackendBarrier at "path" to sync across multiple workers
    (Note that it deletes the path after the sync is done to allow the same path to be reused)
    :param path: path to use for distributed sync (must be shared across workers)
    :param timeout_seconds: how long to wait for nodes to sync
    :param poll_interval: how long to sleep between poll times
    """
    if int(os.environ.get("NUM_NODES", 1)) > 1:
        barrier = FileBackedBarrier(Path(path))
        barrier.wait_barrier(
            activity_id="barrier_token_" + str(os.environ.get('NODE_RANK', 0)),
            expected_activity_ids={"barrier_token_" + str(el) for el in range(0, int(os.environ.get('NUM_NODES', 1)))},
            timeout_s=timeout_seconds,
            poll_interval_s=poll_interval,
        )
