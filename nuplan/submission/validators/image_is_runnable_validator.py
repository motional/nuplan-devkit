import logging

import docker.errors

from nuplan.submission.submission_container import SubmissionContainer
from nuplan.submission.utils.utils import container_name_from_image_name, find_free_port_number
from nuplan.submission.validators.base_submission_validator import BaseSubmissionValidator

logger = logging.getLogger(__name__)


class ImageIsRunnableValidator(BaseSubmissionValidator):
    """Checks if an image is runnable without errors"""

    def validate(self, submission: str) -> bool:
        """
        Checks that the queried image is runnable.
        :param submission: Queried image name
        :return: False if the image is not runnable, or the next validator on the chain if the validation passes
        """
        container_name = container_name_from_image_name(submission)
        submission_container = SubmissionContainer(
            submission_image=submission, container_name=container_name, port=find_free_port_number()
        )
        _ = submission_container.start()

        try:
            submission_container.wait_until_running(timeout=1)
            logger.debug("Image is runnable")

        except TimeoutError:
            logger.error("Image is not runnable")
            self._failing_validator = ImageIsRunnableValidator

            return False

        try:
            submission_container.stop()
        except docker.errors.APIError:
            # It was stopped already
            pass

        return bool(super().validate(submission))
