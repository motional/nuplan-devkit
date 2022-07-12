import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3

from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

logger = logging.getLogger(__name__)


def list_files(source_folder_path: pathlib.Path) -> List[str]:
    """
    List the files present in a directory, including subdirectories
    :param source_folder_path:  Root folder for resources you want to list.
    :return: A string containing relative names of the files.
    """
    paths = []

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
    """Callback for publishing simulation results to AWS at simulation end."""

    def __init__(self, user_id: str, image_id: Optional[str], uploads: Dict[str, Any]):
        """
        Constructor of publisher callback.
        :param user_id: name of the user running the simulation
        :param image_id: image id of the submission, if applicable
        :param uploads: dict containing information on which directories to publish
        """
        submission_prefix = [user_id, image_id] if image_id != "none" else [user_id]

        self.upload_targets: List[UploadConfig] = []

        for name, upload_data in uploads.items():
            if upload_data["upload"]:
                save_path = pathlib.Path(upload_data["save_path"])
                remote_postfix = [save_path.parts[-3], save_path.parts[-1]]
                self.upload_targets.append(
                    UploadConfig(
                        name=name,
                        local_path=save_path,
                        remote_path=pathlib.Path(*submission_prefix, *remote_postfix),  # type: ignore
                    )
                )

    def on_run_simulation_end(self) -> None:
        """
        On reached_end push results to S3 bucket.
        """
        logger.info("Publishing results on S3...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("NUPLAN_SERVER_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("NUPLAN_SERVER_AWS_SECRET_ACCESS_KEY"),
            region_name='us-east-1',
        )
        dest = os.getenv("NUPLAN_SERVER_S3_ROOT_URL")

        for upload_target in self.upload_targets:
            # Get all files to be uploaded from the target
            paths = list_files(upload_target.local_path)

            for path in paths:
                key = str(upload_target.remote_path / path)
                logger.debug(
                    f"Pushing to S3 bucket: {dest}"
                    f"\n\t file: {str(upload_target.local_path.joinpath(path))}"
                    f"\n\t on destination: {key}"
                )
                s3_client.upload_file(str(upload_target.local_path.joinpath(path)), dest, key)

        logger.info("Publishing results on S3... DONE")
