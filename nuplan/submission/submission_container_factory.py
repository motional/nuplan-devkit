from nuplan.submission.submission_container import SubmissionContainer


class SubmissionContainerFactory:
    """Factory for SubmissionContainer"""

    @staticmethod
    def build_submission_container(submission_image: str, container_name: str, port: int) -> SubmissionContainer:
        """
        Builds a SubmissionContainer given submission image, container name and port
        :param submission_image: Name of the Docker image
        :param container_name: Name for the Docker container
        :param port: Port number
        :return: The constructed SubmissionContainer
        """
        return SubmissionContainer(submission_image, container_name, port)
