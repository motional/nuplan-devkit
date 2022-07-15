from typing import Dict

from nuplan.submission.submission_container import SubmissionContainer
from nuplan.submission.submission_container_factory import SubmissionContainerFactory


class SubmissionContainerManager:
    """Class to store created submission containers using a factory."""

    def __init__(self, submission_container_factory: SubmissionContainerFactory):
        """
        :param submission_container_factory: The factory class for the manager
        """
        self.submission_container_factory = submission_container_factory
        self.submission_containers: Dict[str, SubmissionContainer] = {}

    def get_submission_container(self, image: str, container_name: str, port: int) -> SubmissionContainer:
        """
        Returns the queried submission container from the factory, creating it if it's missing
        :param image: Image name
        :param container_name: Container name
        :param port: Port number to open
        :return: The queried submission container
        """
        if container_name not in self.submission_containers:
            self.submission_containers[container_name] = self.submission_container_factory.build_submission_container(
                image, container_name, port
            )

        return self.submission_containers[container_name]
