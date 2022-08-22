import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim.lr_scheduler import OneCycleLR

from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.training.modeling.metrics.planning_metrics import AbstractTrainingMetric
from nuplan.planning.training.modeling.objectives.abstract_objective import aggregate_objectives
from nuplan.planning.training.modeling.objectives.imitation_objective import AbstractObjective
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

logger = logging.getLogger(__name__)


class LightningModuleWrapper(pl.LightningModule):
    """
    Lightning module that wraps the training/validation/testing procedure and handles the objective/metric computation.
    """

    def __init__(
        self,
        model: TorchModuleWrapper,
        objectives: List[AbstractObjective],
        metrics: List[AbstractTrainingMetric],
        batch_size: int,
        optimizer: Optional[DictConfig] = None,
        lr_scheduler: Optional[DictConfig] = None,
        objective_aggregate_mode: str = 'mean',
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.objectives = objectives
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.objective_aggregate_mode = objective_aggregate_mode

        # Validate metrics objectives and model
        model_targets = {builder.get_feature_unique_name() for builder in model.get_list_of_computed_target()}
        for objective in self.objectives:
            for feature in objective.get_list_of_required_target_types():
                assert feature in model_targets, f"Objective target: \"{feature}\" is not in model computed targets!"
        for metric in self.metrics:
            for feature in metric.get_list_of_required_target_types():
                assert feature in model_targets, f"Metric target: \"{feature}\" is not in model computed targets!"

    def _step(self, batch: Tuple[FeaturesType, TargetsType], prefix: str) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets = batch

        predictions = self.forward(features)
        objectives = self._compute_objectives(predictions, targets)
        metrics = self._compute_metrics(predictions, targets)
        loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)

        self._log_step(loss, objectives, metrics, prefix)

        return loss

    def _compute_objectives(self, predictions: TargetsType, targets: TargetsType) -> Dict[str, torch.Tensor]:
        """
        Computes a set of learning objectives used for supervision given the model's predictions and targets.

        :param predictions: model's output signal
        :param targets: supervisory signal
        :return: dictionary of objective names and values
        """
        return {objective.name(): objective.compute(predictions, targets) for objective in self.objectives}

    def _compute_metrics(self, predictions: TargetsType, targets: TargetsType) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        return {metric.name(): metric.compute(predictions, targets) for metric in self.metrics}

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = 'loss',
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(f'loss/{prefix}_{loss_name}', loss)

        for key, value in objectives.items():
            self.log(f'objectives/{prefix}_{key}', value)

        for key, value in metrics.items():
            self.log(f'metrics/{prefix}_{key}', value)

    def training_step(self, batch: Tuple[FeaturesType, TargetsType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'train')

    def validation_step(self, batch: Tuple[FeaturesType, TargetsType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'val')

    def test_step(self, batch: Tuple[FeaturesType, TargetsType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'test')

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,
        Dict[
            str,
            Union[
                torch.optim.Optimizer,
                torch.optim.lr_scheduler._LRScheduler,
            ],
        ],
    ]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        if self.optimizer is None:
            raise RuntimeError("To train, optimizer must not be None.")

        optimizer_dict: Dict[str, Any] = {}

        optimizer: torch.optim.Optimizer = instantiate(
            config=self.optimizer,
            params=self.parameters(),
            lr=self.optimizer.lr,  # Use lr found from lr finder; otherwise use optimizer config
        )

        optimizer_dict['optimizer'] = optimizer
        # Log the optimizer used
        logger.info(f'Using optimizer: {self.optimizer._target_}')

        # Instatiate a learning rate scheduler if it is provided
        if self.lr_scheduler:  # instantiate lr_scheduler according to cfg provided
            lr_scheduler_params: Dict[str, Any] = {}
            if is_target_type(self.lr_scheduler, OneCycleLR):
                # Ensure the initial learning rate used by the LR scheduler is correct by adjusting max_lr
                # This only has to be done for OneCycleLR which overrides the lr in the optimizer provided with max_lr/div_factor

                # Use lr found from lr_find(), otherwise use original optimizer config lr
                self.lr_scheduler.max_lr = self.optimizer.lr

                # ensure lr_scheduler.step() is considered every step, ie every batch (default is every epoch)
                lr_scheduler_params['interval'] = 'step'

                # to ensure that over the course of the training, the learning rate follows 1 cycle, we must call
                # lr_scheduler.step() at a frequency of number of batches per epoch / number of lr_scheduler steps per epoch
                frequency_of_lr_scheduler_step = self.lr_scheduler.epochs

                lr_scheduler_params[
                    'frequency'
                ] = frequency_of_lr_scheduler_step  # number of batches to wait before calling lr_scheduler.step()

                logger.info(f'lr_scheduler.step() will be called every {frequency_of_lr_scheduler_step} batches')

            lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
                config=self.lr_scheduler,
                optimizer=optimizer,
            )
            lr_scheduler_params['scheduler'] = lr_scheduler
            optimizer_dict['lr_scheduler'] = lr_scheduler_params

            # Log the learning rate scheduler used
            logger.info(f'Using lr_scheduler provided: {self.lr_scheduler._target_}')
        else:  # no lr_scheduler provided
            logger.info('Not using any lr_schedulers')

        return optimizer_dict if 'lr_scheduler' in optimizer_dict else optimizer_dict['optimizer']
