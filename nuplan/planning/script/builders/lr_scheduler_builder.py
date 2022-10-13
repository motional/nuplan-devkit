import logging
from typing import Any, Callable, Dict, Optional, cast

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler

from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.utils.lr_schedulers.sequential_lr_scheduler import SequentialLR

logger = logging.getLogger(__name__)

WARM_UP_STRATEGIES = ['constant', 'linear']


def build_lr_scheduler(
    optimizer: Optimizer,
    lr: float,
    warm_up_lr_scheduler_cfg: Optional[DictConfig],
    lr_scheduler_cfg: Optional[DictConfig],
) -> _LRScheduler:
    """
    :param optimizer: Optimizer object
    :param lr: Initial learning rate to be used. If using OneCycleLR, this will be the max learning rate to be used instead.
    :param lr_warm_up: DictConfig for warm up scheduler
    :param lr_scheduler: DictConfig for actual scheduler
    :return: aggregate lr_scheduler
    """
    lr_scheduler_params: Dict[str, Any] = {}

    # Instatiate a learning rate scheduler if it is provided
    if lr_scheduler_cfg is not None:  # instantiate lr_scheduler according to cfg provided
        lr_scheduler_params = _instantiate_main_lr_scheduler(
            optimizer=optimizer, lr_scheduler_cfg=lr_scheduler_cfg, lr=lr
        )
        # Log the learning rate scheduler used
        logger.info(f'Using lr_scheduler provided: {lr_scheduler_cfg._target_}')

    # Instantiate a warm up learning rate scheduler to be used together with the main learning rate scheduler if it is provided
    if warm_up_lr_scheduler_cfg is not None:
        initial_lr = _get_lr_from_optimizer(optimizer)
        lr_scheduler_params = _instantiate_warm_up_lr_scheduler(
            optimizer=optimizer,
            warm_up_lr_scheduler_cfg=warm_up_lr_scheduler_cfg,
            initial_lr=initial_lr,
            lr_scheduler_params=lr_scheduler_params,
        )

    # No lr_scheduler provided
    else:
        logger.info('Not using any lr_schedulers.')

    return lr_scheduler_params


def _get_lr_from_optimizer(optimizer: Optimizer) -> float:
    """
    Gets learning rate from optimizer.
    :param optimizer: Optimizer object.
    :return: Learning rate.
    """
    if len(optimizer.param_groups) == 0:
        raise ValueError('Could not get learning rate.')
    group = optimizer.param_groups[0]
    key = "initial_lr" if "initial_lr" in group else "lr"

    return cast(float, group[key])


def _instantiate_main_lr_scheduler(
    optimizer: Optimizer,
    lr_scheduler_cfg: DictConfig,
    lr: float,
) -> Dict[str, Any]:
    """
    Instantiates the main learning rate scheduler to be used during training.
    :param optimizer: Optimizer used for training.
    :param lr_scheduler_cfg: Learning rate scheduler config
    :param lr: Learning rate to be used. If using OneCycleLR, then this is the maximum learning rate to be reached during training.
    :return: Learning rate scheduler and associated parameters
    """
    lr_scheduler_params: Dict[str, Any] = {}

    if is_target_type(lr_scheduler_cfg, OneCycleLR):
        # Ensure the initial learning rate used by the LR scheduler is correct by adjusting max_lr
        # This only has to be done for OneCycleLR which overrides the lr in the optimizer provided with max_lr/div_factor
        lr_scheduler_cfg.max_lr = lr

        # ensure lr_scheduler.step() is considered every step, ie every batch (default is every epoch)
        lr_scheduler_params['interval'] = 'step'

        # to ensure that over the course of the training, the learning rate follows 1 cycle, we must call
        # lr_scheduler.step() at a frequency of number of batches per epoch / number of lr_scheduler steps per epoch
        frequency_of_lr_scheduler_step = lr_scheduler_cfg.epochs

        lr_scheduler_params[
            'frequency'
        ] = frequency_of_lr_scheduler_step  # number of batches to wait before calling lr_scheduler.step()

        logger.info(f'lr_scheduler.step() will be called every {frequency_of_lr_scheduler_step} batches')

    lr_scheduler: _LRScheduler = instantiate(
        config=lr_scheduler_cfg,
        optimizer=optimizer,
    )
    lr_scheduler_params['scheduler'] = lr_scheduler

    return lr_scheduler_params


def _instantiate_warm_up_lr_scheduler(
    optimizer: Optimizer,
    warm_up_lr_scheduler_cfg: DictConfig,
    initial_lr: float,
    lr_scheduler_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Instantiates the warm up learning rate scheduler to be used during training.
    :param optimizer: Optimizer used for training.
    :param warm_up_lr_scheduler_cfg: Learning rate scheduler config for warm_up phase
    :param initial_lr: Initial learning rate. To be scaled down further during warm_up phase.
    :param lr: Learning rate to be used. If using OneCycleLR, then this is the maximum learning rate to be reached during training.
    :return: Learning rate scheduler and associated parameters
    """
    # Check if also using another lr_scheduler as the main lr_scheduler
    using_main_lr_scheduler = 'scheduler' in lr_scheduler_params

    if using_main_lr_scheduler:  # if using another scheduler as the main scheduler

        warm_up_lr_scheduler = instantiate(
            config=warm_up_lr_scheduler_cfg,
            optimizer=optimizer,
            lr_lambda=instantiate(config=warm_up_lr_scheduler_cfg.lr_lambda, final_div_factor=1.0),
        )

        lr_schedulers = [warm_up_lr_scheduler, lr_scheduler_params['scheduler']]

        # Aggregate lr_schedulers into a combined lr_scheduler where the warm up scheduler
        # is called during the warm up phase before the main scheduler is called.
        sequential_lr_scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=lr_schedulers,
            milestones=[warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps],
        )

        lr_scheduler_params['scheduler'] = sequential_lr_scheduler
        logger.info(
            f'Added Warm up learning rate scheduler before main scheduler with {warm_up_lr_scheduler_cfg.lr_lambda.warm_up_strategy} strategy.'
        )

    else:  # No main learning rate scheduler is used. Only a warm up learning rate scheduler is specified.
        warm_up_lr_scheduler = instantiate(
            config=warm_up_lr_scheduler_cfg,
            optimizer=optimizer,
            lr_lambda=instantiate(
                config=warm_up_lr_scheduler_cfg.lr_lambda,
                final_div_factor=1.0,
            ),
        )

        # Adjust initial learning rate
        warm_up_phase_initial_lr = initial_lr / warm_up_lr_scheduler_cfg.lr_lambda.warm_up_steps
        for group in optimizer.param_groups:
            group['initial_lr'] = warm_up_phase_initial_lr

        lr_scheduler_params['scheduler'] = warm_up_lr_scheduler

        logger.info(
            f'Using Warm up learning rate scheduler with {warm_up_lr_scheduler_cfg.lr_lambda.warm_up_strategy} strategy.'
        )

    return lr_scheduler_params


def get_warm_up_lr_scheduler_func(
    warm_up_steps: int, warm_up_strategy: str, final_div_factor: float = 1.0
) -> Callable[..., float]:
    """
    Gets the lambda function for the warm up learning rate scheduler.
    :param warm_up_steps: Number of steps allocated to warm up scheduler.
    :param warm_up_strategy: Strategy for the warm up phase.
    :param final_div_factor: Amount to scale the initial learning rate.
    :return: Function for warm up learning rate scheduler.
    """

    def _get_linear_warm_up_func(step: int) -> float:
        """
        Returns multiplicative factor that linearly increases learning rate during warm_up phase.
        :param step: Number of calls made to the warm_up learning rate scheduler so far.
        :return: Multiplicative factor to scale the initial learning rate by.
        """
        return step / warm_up_steps * final_div_factor if step <= warm_up_steps else 1.0

    def _get_constant_warm_up_func(step: int) -> float:
        """
        Returns multiplicative factor that scales down learning rate to a lower but constant value during warm_up phase.
        :param step: Number of calls made to the warm_up learning rate scheduler so far.
        :return: Multiplicative factor to scale the initial learning rate by.
        """
        return 1 / warm_up_steps * final_div_factor if step <= warm_up_steps else 1.0

    strategy_name_mapping: Dict[str, Callable[..., float]] = {
        'linear': _get_linear_warm_up_func,
        'constant': _get_constant_warm_up_func,
    }

    if warm_up_strategy not in strategy_name_mapping:
        raise ValueError(
            f'Warm_up strategy {warm_up_strategy} is not currently supported. Supported learning rate warm up strategies: {list(strategy_name_mapping.keys())}'
        )

    return strategy_name_mapping[warm_up_strategy]
