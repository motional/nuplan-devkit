import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3

from nuplan.common.utils.s3_utils import get_s3_client
from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

logger = logging.getLogger(__name__)


def list_files(source_folder_path: pathlib.Path) -> List[str]:
    """
    List the files present in a directory, including subdirectories
    :param source_folder_path:  Root folder for resources you want to list.
    :return: A string containing relative names of the files.
    """
    paths = []

    if source_folder_path.is_file():
        logger.info("Provided path was a file, returning filename only.")
        return [source_folder_path.parts[-1]]

    for file_path in source_folder_path.rglob("*"):
        if file_path.is_dir():
            continue
        str_file_path = str(file_path)
        str_file_path = str_file_path.replace(f'{str(source_folder_path)}/', "")
        paths.append(str_file_path)

    return paths


@dataclass
class UploadConfig:
    """Config specifying files to be uploaded and their target paths."""

    name: str
    local_path: pathlib.Path
    remote_path: pathlib.Path


class PublisherCallback(AbstractMainCallback):
    """Callback publishing data to S3"""

    def __init__(
        self,
        uploads: Dict[str, Any],
        s3_client: Optional[boto3.client],
        s3_bucket: str,
        remote_prefix: Optional[List[str]],
    ):
        """
        Construct publisher callback, responsible to publish results of simulation, image validation and result aggregation
        :param uploads: dict containing information on which directories to publish
        """
        self._s3_client = s3_client
        if self._s3_client is None:
            self._s3_client = get_s3_client()
        self._s3_bucket = s3_bucket.strip('s3://') if s3_bucket.startswith('s3://') else s3_bucket
        self._remote_prefix: List[str] = remote_prefix or ['/']

        self._upload_targets: List[UploadConfig] = []

        for name, upload_data in uploads.items():
            if upload_data["upload"]:
                save_path = pathlib.Path(upload_data["save_path"])
                remote_path = pathlib.Path(upload_data.get("remote_path") or "")

                self._upload_targets.append(
                    UploadConfig(
                        name=name,
                        local_path=save_path,
                        remote_path=pathlib.Path(*self._remote_prefix) / remote_path,
                    )
                )

    def on_run_simulation_end(self) -> None:
        """
        On reached_end push results to S3 bucket.
        """
        logger.info("Publishing results on S3...")

        for upload_target in self._upload_targets:
            # Get all files to be uploaded from the target
            paths = list_files(upload_target.local_path)

            for path in paths:
                key = str(upload_target.remote_path / path)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Pushing to S3 bucket: {self._s3_bucket}"
                        f"\n\t file: {str(upload_target.local_path.joinpath(path))}"
                        f"\n\t on destination: {key}"
                    )

                local_target = upload_target.local_path
                if not local_target.is_file():
                    local_target = local_target.joinpath(path)

                self._s3_client.upload_file(str(local_target), self._s3_bucket, key)  # type: ignore

        logger.info("Publishing results on S3... DONE")
