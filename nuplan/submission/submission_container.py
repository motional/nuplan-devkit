from __future__ import annotations

import logging
import os
from typing import Any

import docker
from docker.errors import NotFound

from nuplan.common.utils.helpers import keep_trying

logger = logging.getLogger(__name__)


class SubmissionContainer:
    """Class handling a submission Docker container"""

    def __init__(self, submission_image: str, container_name: str, port: int):
        """
        :param submission_image: Name of the docker image of the submission
        :param container_name: Name for the container to be run
        :param port: Port number to be used for communication
        """
        self.submission_image = submission_image
        self.container_name = container_name
        self.port = port
        self.client: docker.client.DockerClient | None = None

    def __del__(self) -> None:
        """Stop the running container when the destructor is called."""
        self.stop()

    def start(self, cpus: str = "0,1", gpus: list[str] | None = None) -> Any:
        """
        Starts the submission container given a docker client, and the submission details. It exposes the specified
        port, and volume mounts (read only) the data directory to make it available to the container.
        :param cpus: CPUs to be used by the submission container
        :param gpus: GPUs to be used by the submission container
        """
        if gpus is None:
            gpus = ["0"]
        self.client = docker.from_env()

        self.stop()

        ports = {f'{str(self.port)}': self.port}
        self.client.containers.run(
            self.submission_image,
            name=self.container_name,
            detach=True,
            ports=ports,
            tty=True,
            environment={'SUBMISSION_CONTAINER_PORT': str(self.port)},
            device_requests=[docker.types.DeviceRequest(device_ids=gpus, capabilities=[['gpu']])],
            cpuset_cpus=cpus,
            volumes={os.getenv('NUPLAN_DATA_ROOT', "~/nuplan/dataset"): {'bind': '/data/sets/nuplan', 'mode': 'ro'}},
        )
        logging.debug(f"Started submission container with image: {self.submission_image} with port: {self.port}")

        return self.client.containers.get(self.container_name)

    def stop(self) -> None:
        """Checks if the submission container is running, if it is it stops and removes it."""
        try:
            container = self.client.containers.get(self.container_name)  # type: ignore
        except NotFound:
            pass
        else:

            logging.debug("Stopping and removing pre-existing container")
            try:
                container.kill()
            except docker.errors.APIError:
                # If it was not running this would throw
                pass
            container.remove()

    def wait_until_running(self, timeout: float = 3) -> None:
        """
        Waits until a container is running until timeout.
        :param timeout: timeout in seconds
        """

        def is_running(manager: SubmissionContainer) -> bool:
            """
            Checks if the container is running
            :param manager: The container manager
            :returns: True if the container is in running state
            """
            return bool(
                manager.client.api.inspect_container(manager.container_name)["State"]["Status"]  # type: ignore
                == "running"
            )

        keep_trying(is_running, [self], {}, (docker.errors.NotFound,), timeout)
