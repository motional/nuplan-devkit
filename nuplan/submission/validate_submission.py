import logging
from typing import Optional, Type

from nuplan.submission.validators.base_submission_validator import AbstractSubmissionValidator, BaseSubmissionValidator
from nuplan.submission.validators.image_exists_validator import ImageExistsValidator
from nuplan.submission.validators.image_is_runnable_validator import ImageIsRunnableValidator
from nuplan.submission.validators.submission_computes_trajectory_validator import SubmissionComputesTrajectoryValidator


def validate_submission(
    image: str, validator: BaseSubmissionValidator
) -> tuple[bool, Optional[Type[AbstractSubmissionValidator]]]:
    """
    Calls the chain of validators on one image.
    :param image: The query docker image
    :param validator: The chain of validators
    :return: A tuple with two possible values:
        (True, None) If the image is valid
        (False, Failing validator type) if image is deemed invalid by a validator on the chain
    """
    image_is_valid = validator.validate(image)

    return bool(image_is_valid), validator.failing_validator


if __name__ == '__main__':

    logger = logging.getLogger("validate_submission")
    logging.basicConfig(level=logging.INFO)

    submission_validator = BaseSubmissionValidator()
    image_exists = ImageExistsValidator()
    image_is_runnable = ImageIsRunnableValidator()
    submission_computes_trajectory = SubmissionComputesTrajectoryValidator()

    submission_validator.set_next(image_exists).set_next(image_is_runnable).set_next(submission_computes_trajectory)

    submission_image = "test-contestant/nuplan-test-submission:latest"
    valid = validate_submission(submission_image, submission_validator)

    if not valid[0]:
        logger.error(f"Image is not valid: \n {submission_image} failing {valid[1]}")
        exit(1)

    logger.info("Image is valid, pushing to queue...")
    # TODO: push image
