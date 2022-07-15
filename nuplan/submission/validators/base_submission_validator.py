from __future__ import annotations

from nuplan.submission.validators.abstract_submission_validator import AbstractSubmissionValidator


class BaseSubmissionValidator(AbstractSubmissionValidator):
    """Base validator for submission validation."""

    def __init__(self) -> None:
        """Constructor, sets next validator and failing validator to none"""
        self._next_validator: AbstractSubmissionValidator | None = None
        self._failing_validator: type[AbstractSubmissionValidator] | None = None

    def set_next(self, validator: AbstractSubmissionValidator) -> AbstractSubmissionValidator:
        """
        Sets the next validator in the chain
        :param validator: The next validator
        :return: The set validator
        """
        self._next_validator = validator
        return validator

    def validate(self, submission: str) -> bool:
        """
        Validates the given submission.
        :param submission: Query submission
        :return: True, if no validator is present, otherwise the next validator validate method output.
        """
        if self._next_validator:
            return bool(self._next_validator.validate(submission))

        return True

    @property
    def failing_validator(self) -> type[AbstractSubmissionValidator] | None:
        """
        Getter for the failing validator
        :return: the failing validator
        """
        return self._failing_validator
