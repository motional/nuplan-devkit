import logging
import pathlib
from typing import List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.serialization_callback import SerializationCallback
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.timing_callback import TimingCallback
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool, WorkerResources
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

logger = logging.getLogger(__name__)


def build_callbacks_worker(cfg: DictConfig) -> Optional[WorkerPool]:
    """
    Builds workerpool for callbacks.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Workerpool, or None if we'll run without one.
    """
    if not is_target_type(cfg.worker, Sequential) or cfg.disable_callback_parallelization:
        return None

    if cfg.number_of_cpus_allocated_per_simulation not in [None, 1]:
        raise ValueError("Expected `number_of_cpus_allocated_per_simulation` to be set to 1 with Sequential worker.")

    max_workers = min(
        WorkerResources.current_node_cpu_count() - (cfg.number_of_cpus_allocated_per_simulation or 1),
        cfg.max_callback_workers,
    )
    callbacks_worker_pool = SingleMachineParallelExecutor(use_process_pool=True, max_workers=max_workers)
    return callbacks_worker_pool


def build_simulation_callbacks(
    cfg: DictConfig, output_dir: pathlib.Path, worker: Optional[WorkerPool] = None
) -> List[AbstractCallback]:
    """
    Builds callback.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param output_dir: directory for all experiment results.
    :param worker: to run certain callbacks in the background (everything runs in main process if None).
    :return: List of callbacks.
    """
    logger.info('Building AbstractCallback...')
    callbacks = []
    for config in cfg.callback.values():
        if is_target_type(config, SerializationCallback):
            callback: SerializationCallback = instantiate(config, output_directory=output_dir)
        elif is_target_type(config, TimingCallback):
            tensorboard = torch.utils.tensorboard.SummaryWriter(log_dir=output_dir)
            callback = instantiate(config, writer=tensorboard)
        elif is_target_type(config, SimulationLogCallback) or is_target_type(config, MetricCallback):
            # SimulationLogCallback and MetricCallback store state (futures) from each runner, so they are initialized
            # in the simulation builder
            continue
        else:
            callback = instantiate(config)
        validate_type(callback, AbstractCallback)
        callbacks.append(callback)
    logger.info(f'Building AbstractCallback: {len(callbacks)}...DONE!')
    return callbacks
