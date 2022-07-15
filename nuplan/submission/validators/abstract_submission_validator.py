from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractSubmissionValidator(ABC):
    """Abstract class for submission validators."""

    @abstractmethod
    def set_next(self, validator: AbstractSubmissionValidator) -> AbstractSubmissionValidator:
        """
        Sets the next validator for the chain of validation of the submission.
        :param validator: The validator to be added after self
        :return: the added validator
        """
        pass

    @abstractmethod
    def validate(self, submission: str) -> bool:
        """
        Validates whether an image passes the chain of checks specified in the single validators.
        :param submission: The query submission
        :return: Whether all checks are passed
        """
        pass
