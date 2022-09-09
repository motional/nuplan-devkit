from abc import ABC, abstractmethod
from typing import List

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class AbstractSplitter(ABC):
    """
    Abstract splitter class for splitting database to lists of samples for each of train/val/test sets.
    """

    @abstractmethod
    def get_train_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        """
        Extracts a list of samples to be used for training.

        :param scenarios: candidate logs containing the samples
        :return: list of selected samples
        """
        pass

    @abstractmethod
    def get_val_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        """
        Extracts a list of samples to be used for validation.

        :param scenarios: candidate scenarios containing the samples
        :return: list of selected samples
        """
        pass

    @abstractmethod
    def get_test_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        """
        Extracts a list of samples to be used for testing.

        :param scenarios: candidate scenarios containing the samples
        :return: list of selected samples
        """
        pass
