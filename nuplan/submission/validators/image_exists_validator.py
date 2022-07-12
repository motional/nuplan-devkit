import logging

import docker

from nuplan.submission.validators.base_submission_validator import BaseSubmissionValidator

logger = logging.getLogger(__name__)


class ImageExistsValidator(BaseSubmissionValidator):
    """SubmissionValidator that checks if an image is present"""

    def validate(self, submission: str) -> bool:
        """
        Checks that the queried image exists locally.
        :param submission: Queried image name
        :return: False if the image is not available, or the next validator on the chain if the validation passes
        """
        client = docker.from_env()
        tags = []

        for image in client.images.list(name=submission):
            tags.extend(image.tags)

        if submission not in tags:
            self._failing_validator = ImageExistsValidator
            logger.error("Image doesn't exist")

            return False

        logger.debug("Image exists")

        return bool(super().validate(submission))
