import unittest
from unittest.mock import Mock

import numpy as np
import pytorch_lightning as pl

from nuplan.planning.training.callbacks.stepwise_augmentation_scheduler import (
    StepwiseAugmentationAttributeScheduler,
    StepwiseAugmentationProbabilityScheduler,
)
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
)


class TestStepwiseAugmentationSheduler(unittest.TestCase):
    """Test scenario scoring callback"""

    def setUp(self) -> None:
        """Set up test case."""
        super().setUp()
        self.max_augment_prob = 0.8
        self.pct_time_increasing = 0.5

        self.max_aug_attribute_pct_increase = 0.2
        self.initial_augment_prob = 0.5
        self.milestones = [0.25, 0.5, 0.75, 1.0]
        self.cur_step = 1
        self.total_steps = 2

        # augmentor properties
        self.mock_trajectory_length = 12
        self.mock_dt = 0.5
        self.mock_mean = [1.0, 0.0, 0.0]
        self.mock_std = [1.0, 1.0, 0.5]
        self.mock_low = [0.0, -1.0, -0.5]
        self.mock_high = [1.0, 1.0, 0.5]
        self.mock_augmentation_probability = 0.5
        self.mock_use_uniform_noise = False

    def test_scale_augmentor(self) -> None:
        """
        Test scale_augmentor function.
        """
        augmentation_attribute_scheduler = StepwiseAugmentationAttributeScheduler(
            self.max_aug_attribute_pct_increase, self.pct_time_increasing, 'linear', self.milestones
        )

        augmentation_probability_scheduler = StepwiseAugmentationProbabilityScheduler(
            self.max_augment_prob,
            self.pct_time_increasing,
            'linear',
            self.milestones,
        )

        # Mock augmentor
        mock_augmentor = Mock(AbstractAugmentor)
        mock_augmentor._random_offset_generator = GaussianNoise(self.mock_mean, self.mock_std)
        mock_augmentor._augment_prob = self.mock_augmentation_probability

        # Mock augmentor methods and properties
        mock_augmentor.__name__ = Mock(return_value='mock_augmentor')
        mock_augmentor.augmentation_probability = ParameterToScale(
            param=self.mock_augmentation_probability, param_name='_augment_prob', scaling_direction=ScalingDirection.MAX
        )
        mock_augmentor.get_schedulable_attributes = mock_augmentor._random_offset_generator.get_schedulable_attributes()

        # Scale augmentor
        augmentation_attribute_scheduler._scale_augmentor(
            mock_augmentor,
            self.cur_step,
            self.total_steps,
        )
        augmentation_probability_scheduler._scale_augmentor(
            mock_augmentor,
            self.cur_step,
            self.total_steps,
        )
        expected_mean = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_mean)
        expected_std = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_std)
        expected_augmentation_probability = self.max_augment_prob
        self.assertEqual(mock_augmentor._augment_prob, expected_augmentation_probability)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, expected_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, expected_std))

    def test_handle_scheduling(self) -> None:
        """
        Test _handle_scheduling function to ensure scaling doesn't happen on non milestone steps.
        """
        augmentation_attribute_scheduler = StepwiseAugmentationAttributeScheduler(
            self.max_aug_attribute_pct_increase, self.pct_time_increasing, 'milestones', self.milestones
        )

        augmentation_probability_scheduler = StepwiseAugmentationProbabilityScheduler(
            self.max_augment_prob,
            self.pct_time_increasing,
            'milestones',
            self.milestones,
        )

        # Mock augmentor
        mock_augmentor = Mock(AbstractAugmentor)
        mock_augmentor._random_offset_generator = GaussianNoise(self.mock_mean, self.mock_std)
        mock_augmentor._augment_prob = self.mock_augmentation_probability

        # Mock augmentor methods and properties
        mock_augmentor.__name__ = Mock(return_value='mock_augmentor')
        mock_augmentor.augmentation_probability = ParameterToScale(
            param=self.mock_augmentation_probability, param_name='_augment_prob', scaling_direction=ScalingDirection.MAX
        )
        mock_augmentor.get_schedulable_attributes = mock_augmentor._random_offset_generator.get_schedulable_attributes()

        # Mock pl trainer
        mock_trainer = Mock(pl.Trainer)
        mock_trainer.datamodule = Mock()
        mock_trainer.datamodule._train_set._augmentors = [mock_augmentor]

        # Scale augmentor with non-milestone step, at 10% of the training
        non_milestone_cur_step = 0.1
        pct_progress = round(non_milestone_cur_step / (self.total_steps * self.pct_time_increasing), 2)
        augmentation_attribute_scheduler._handle_scheduling(
            mock_trainer, non_milestone_cur_step, self.total_steps, pct_progress
        )
        augmentation_probability_scheduler._handle_scheduling(
            mock_trainer, non_milestone_cur_step, self.total_steps, pct_progress
        )

        # Expect that there is no change in values since it is not on a milestone
        self.assertEqual(mock_augmentor._augment_prob, self.initial_augment_prob)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, self.mock_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, self.mock_std))

        # Scale augmentor again  with milestone step, at 100% of the training
        pct_progress = round(self.cur_step / (self.total_steps * self.pct_time_increasing), 2)
        augmentation_attribute_scheduler._handle_scheduling(mock_trainer, self.cur_step, self.total_steps, pct_progress)
        augmentation_probability_scheduler._handle_scheduling(
            mock_trainer, self.cur_step, self.total_steps, pct_progress
        )
        # Expect that there is change in values since it is on a milestone
        expected_mean = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_mean)
        expected_std = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_std)
        expected_augmentation_probability = self.max_augment_prob
        self.assertEqual(mock_augmentor._augment_prob, expected_augmentation_probability)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, expected_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, expected_std))

    def test_on_batch_end_milestones(self) -> None:
        """
        Test on_batch_end function to ensure scaling doesn't happen after scheduling is completed using milestones strategy.
        """
        augmentation_attribute_scheduler = StepwiseAugmentationAttributeScheduler(
            self.max_aug_attribute_pct_increase, self.pct_time_increasing, 'milestones', self.milestones
        )

        augmentation_probability_scheduler = StepwiseAugmentationProbabilityScheduler(
            self.max_augment_prob,
            self.pct_time_increasing,
            'milestones',
            self.milestones,
        )

        # Mock augmentor
        mock_augmentor = Mock(AbstractAugmentor)
        mock_augmentor._random_offset_generator = GaussianNoise(self.mock_mean, self.mock_std)
        mock_augmentor._augment_prob = self.mock_augmentation_probability

        # Mock augmentor methods and properties
        mock_augmentor.__name__ = Mock(return_value='mock_augmentor')
        mock_augmentor.augmentation_probability = ParameterToScale(
            param=self.mock_augmentation_probability, param_name='_augment_prob', scaling_direction=ScalingDirection.MAX
        )
        mock_augmentor.get_schedulable_attributes = mock_augmentor._random_offset_generator.get_schedulable_attributes()

        # Mock pl trainer
        mock_trainer = Mock(pl.Trainer)
        mock_trainer.max_epochs = 2
        mock_trainer.num_training_batches = 1
        mock_trainer.datamodule = Mock()
        mock_trainer.datamodule._train_set._augmentors = [mock_augmentor]

        # Mock pl lightning module
        mock_module = Mock(pl.LightningModule)

        # After this step, augmentors should have completed scheduling.
        mock_trainer.global_step = 0  # At the end of 1st training batch
        augmentation_attribute_scheduler.on_batch_end(mock_trainer, mock_module)
        augmentation_probability_scheduler.on_batch_end(mock_trainer, mock_module)

        # Expect that the final desired values have been reached in the augmentor
        expected_mean = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_mean)
        expected_std = (1 + self.max_aug_attribute_pct_increase) * np.asarray(self.mock_std)
        expected_augmentation_probability = self.max_augment_prob
        self.assertEqual(mock_augmentor._augment_prob, expected_augmentation_probability)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, expected_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, expected_std))

        # Run on_batch_end again after scheduling has completed and expect that nothing was scaled
        mock_trainer.global_step = 1  # At the end of last training batch
        augmentation_attribute_scheduler.on_batch_end(mock_trainer, mock_module)
        augmentation_probability_scheduler.on_batch_end(mock_trainer, mock_module)
        self.assertEqual(mock_augmentor._augment_prob, expected_augmentation_probability)
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.mean, expected_mean))
        self.assertTrue(np.allclose(mock_augmentor._random_offset_generator.std, expected_std))


if __name__ == '__main__':
    unittest.main()
