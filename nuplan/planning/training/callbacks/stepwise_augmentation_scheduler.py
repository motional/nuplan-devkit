from abc import abstractmethod
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl

from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import ParameterToScale, ScalingDirection


class SchedulingStrategies(Enum):
    """Enum class for scheduling strategies."""

    linear = auto()  # Scales augmentor attributes every training step
    milestones = auto()  # Scales augmentor attributes only at specific milestones during training


class AbstractAugmentationScheduler(pl.Callback):
    """Augmentation scheduler callback class."""

    def __init__(self, pct_time_increasing: float, scheduling_strategy: str, milestones: Optional[List[float]] = None):
        """
        Initializes Augmentation Scheduler Callback properties.
        :param pct_time_increasing: Percentage of the time spent increasing.
        :param scheduling strategy: Strategy for scheduling the scaling of augmentor properties during training.
        :param milestones: List of milestones for scheduling augmentation. Eg, given [0.25, 0.5, 0.75, 1.0], we will increase the augmentation when 25%, 50%, 75% of the training has completed.
        """
        self._pct_time_increasing = pct_time_increasing
        assert (
            0.0 <= self._pct_time_increasing <= 1.0
        ), 'Error, percentage of time spent increasing augmentation must be between 0 and 1.'

        self._milestones = (
            sorted(round(milestone, 2) for milestone in list(set(milestones))) if milestones is not None else []
        )
        if len(self._milestones) > 0:  # If milestones are provided
            assert all(
                [0.0 <= milestone <= 1.0 for milestone in self._milestones]
            ), 'Error milestones must all be between 0 and 1.'
            assert self._milestones[-1] == 1.0, 'Error, milestones must include 1 as the final value.'

        self._scheduling_strategy = SchedulingStrategies[scheduling_strategy]
        assert (
            self._scheduling_strategy in SchedulingStrategies
        ), f"Error, scheduling strategy {self._scheduling_strategy} is not supported."
        if self._scheduling_strategy == SchedulingStrategies.milestones:
            assert (
                len(self._milestones) > 0
            ), 'Error, schdeuling strategy "milestones" requires milstones to be specified.'

        self._initial_augmentor_attributes: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(lambda: Any))
        self._completed_scheduling = False

    @abstractmethod
    def _scale_augmentor(self, augmentor: AbstractAugmentor, cur_step: int, total_steps: int) -> None:
        """
        Scales augmentator properties:
        :param augmentor: Abstract augmentor.
        :param cur_step: Current training step.
        :param total_steps: Total number of training steps.
        """
        pass

    def _scale_augmentor_property(
        self, initial_attr: ParameterToScale, pct_increase: float, cur_step: int, total_steps: int
    ) -> Any:
        """
        Scales the augmentor property.
        :param initial_attr: ParameterToScale object for augmentor property.
        :param pct_increase: Desired percentage to increase property by.
        :param cur_step: Current training step.
        :param total_steps: Total number of training steps.
        :return: Scaled augmentor property.
        """
        pct_progress = cur_step / (total_steps * self._pct_time_increasing)

        scaling_direction = initial_attr.scaling_direction
        assert (
            scaling_direction in ScalingDirection
        ), f'Error, scaling direction {scaling_direction} is not supported. Supported scaling directions: {ScalingDirection}'

        scaling_sign = 1 if scaling_direction == ScalingDirection.MAX else -1
        scaled_attr = (1 + scaling_sign * pct_increase * pct_progress) * initial_attr.param
        return scaled_attr

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        The end of each batch.
        :param trainer: Pytorch Lightning Trainer.
        :param pl_module: Pytorch Lightning Module.
        """
        max_epochs = trainer.max_epochs
        total_steps = max_epochs * trainer.num_training_batches
        cur_step = trainer.global_step + 1  # step starts from 0
        pct_progress = round(cur_step / (total_steps * self._pct_time_increasing), 2)  # Round to 2 decimal points

        # if pct_progress is already > 1, stop scheduling
        self._completed_scheduling = pct_progress > 1

        if not self._completed_scheduling:
            self._handle_scheduling(trainer, cur_step, total_steps, pct_progress)

    def _handle_scheduling(self, trainer: pl.Trainer, cur_step: int, total_steps: int, pct_progress: float) -> None:
        """Function to handle scheduling according to sheduling strategy"""
        if self._scheduling_strategy != SchedulingStrategies.milestones:

            for augmentor in trainer.datamodule._train_set._augmentors:
                self._scale_augmentor(augmentor, cur_step, total_steps)
        else:  # If scaling by miletones, only call the _scale_augmentor function on the milestones
            if pct_progress in self._milestones:
                self._milestones.remove(pct_progress)

                for augmentor in trainer.datamodule._train_set._augmentors:
                    self._scale_augmentor(augmentor, cur_step, total_steps)

            self._completed_scheduling = len(self._milestones) == 0


class StepwiseAugmentationProbabilityScheduler(AbstractAugmentationScheduler):
    """Callback class that scales Augmentors."""

    def __init__(
        self,
        max_augment_prob: float,
        pct_time_increasing: float,
        scheduling_strategy: str,
        milestones: Optional[List[float]] = None,
    ):
        """
        Initializes Augmentation Scheduler Callback properties.
        :param max_augment_prob: Maximum probability of augmentation.
        :param pct_time_increasing: Percentage of the time spent increasing.
        :param scheduling strategy: Strategy for scheduling the scaling of augmentor properties during training.
        :param milestones: List of milestones for scheduling augmentation. Eg, given [0.25, 0.5, 0.75, 1.0], we will increase the augmentation when 25%, 50%, 75% of the training has completed.
        """
        super().__init__(pct_time_increasing, scheduling_strategy, milestones)

        self.max_augment_prob = max_augment_prob
        assert 0.0 <= self.max_augment_prob <= 1.0, 'Error max augmentation probability must be between 0 and 1'

    def _scale_augmentor(self, augmentor: AbstractAugmentor, cur_step: int, total_steps: int) -> None:
        """
        Scales augmentor properties.
        :param augmentor: Abstract augmentor.
        :param cur_step: Current training step.
        :param total_steps: Total number of training steps.
        """
        # Store initial values for augmentor properties
        augmentor_name = type(augmentor).__name__
        if augmentor_name not in self._initial_augmentor_attributes:  # Initialize initial values
            # Store initial augmentation probability
            aug_prob = augmentor.augmentation_probability
            self._initial_augmentor_attributes[augmentor_name][aug_prob.param_name] = aug_prob

        # Scale augmentation probability
        for param_name, param_to_sale_obj in self._initial_augmentor_attributes[augmentor_name].items():
            pct_increase = (
                self.max_augment_prob / self._initial_augmentor_attributes[augmentor_name][param_name].param - 1
            )
            scaled_augment_prob = self._scale_augmentor_property(
                initial_attr=param_to_sale_obj,
                pct_increase=pct_increase,
                cur_step=cur_step,
                total_steps=total_steps,
            )
            setattr(augmentor, param_name, scaled_augment_prob)


class StepwiseAugmentationAttributeScheduler(AbstractAugmentationScheduler):
    """Callback class that scales Noise parameters."""

    def __init__(
        self,
        max_aug_attribute_pct_increase: float,
        pct_time_increasing: float,
        scheduling_strategy: str,
        milestones: Optional[List[float]] = None,
    ):
        """
        Initializes Augmentation Scheduler Callback properties.
        :param max_aug_attribute_pct_increase: Percentage increase in augmentor attributes to be reached at end of training.
        :param pct_time_increasing: Percentage of the time spent increasing.
        :param scheduling strategy: Strategy for scheduling the scaling of augmentor attributes during training.
        :param milestones: List of milestones for scheduling augmentation. Eg, given [0.25, 0.5, 0.75, 1.0], we will increase the augmentation when 25%, 50%, 75% of the training has completed.
        """
        super().__init__(pct_time_increasing, scheduling_strategy, milestones)
        self.max_aug_attribute_pct_increase = max_aug_attribute_pct_increase

    def _scale_augmentor(self, augmentor: AbstractAugmentor, cur_step: int, total_steps: int) -> None:
        """
        Scales augmentor attributes.
        :param augmentor: Abstract augmentor.
        :param cur_step: Current training step.
        :param total_steps: Total number of training steps.
        """
        # Store initial augmentor property names and ParameterToScale objects for each augmentor property
        augmentor_name = type(augmentor).__name__
        if augmentor_name not in self._initial_augmentor_attributes:
            # Store initial values and scaling directions for augmentor attributes
            for param_to_scale in augmentor.get_schedulable_attributes:
                self._initial_augmentor_attributes[augmentor_name][param_to_scale.param_name] = param_to_scale

        # Scale each augmentor property according to scaling direction
        for param_name, param_to_sale_obj in self._initial_augmentor_attributes[augmentor_name].items():

            scaled_attr = self._scale_augmentor_property(
                initial_attr=param_to_sale_obj,
                pct_increase=self.max_aug_attribute_pct_increase,
                cur_step=cur_step,
                total_steps=total_steps,
            )
            setattr(augmentor._random_offset_generator, param_name, scaled_attr)
