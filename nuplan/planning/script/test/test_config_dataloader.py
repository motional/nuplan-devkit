import os
import tempfile
import unittest

import pytorch_lightning as pl
import torch.utils.data
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.training_builder import build_lightning_datamodule
from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.script.builders.worker_pool_builder import build_worker

CONFIG_NAME = 'default_training'


class TestDataLoader(unittest.TestCase):
    """
    Tests data loading functionality
    """

    def setUp(self) -> None:
        """Setup hydra config."""
        seed = 10
        pl.seed_everything(seed, workers=True)

        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/training/')
        self.group = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.group.name, 'cache_path')

    def tearDown(self) -> None:
        """Remove temporary folder."""
        self.group.cleanup()

    @staticmethod
    def validate_cfg(cfg: DictConfig) -> None:
        """Validate hydra config."""
        update_config_for_training(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.scenario_filter.limit_total_scenarios = 0.001
        cfg.data_loader.datamodule.train_fraction = 1.0
        cfg.data_loader.datamodule.val_fraction = 1.0
        cfg.data_loader.datamodule.test_fraction = 1.0
        cfg.data_loader.params.batch_size = 2
        cfg.data_loader.params.num_workers = 2
        cfg.data_loader.params.pin_memory = False
        OmegaConf.set_struct(cfg, True)

    @staticmethod
    def _iterate_dataloader(dataloader: torch.utils.data.DataLoader) -> None:
        """
        Iterate a fixed number of batches of the dataloader.
        :param dataloader: Data loader to iterate.
        """
        num_batches = 5
        dataloader_iter = iter(dataloader)
        iterations = min(len(dataloader), num_batches)

        for _ in range(iterations):
            next(dataloader_iter)

    def _run_dataloader(self, cfg: DictConfig) -> None:
        """
        Test that the training dataloader can be iterated without errors.
        :param cfg: Hydra config.
        """
        worker = build_worker(cfg)
        lightning_module_wrapper = build_torch_module_wrapper(cfg.model)
        datamodule = build_lightning_datamodule(cfg, worker, lightning_module_wrapper)
        datamodule.setup('fit')
        datamodule.setup('test')

        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.test_dataloader()

        for dataloader in [train_dataloader, val_dataloader]:
            assert len(dataloader) > 0
            self._iterate_dataloader(dataloader)

        self._iterate_dataloader(test_dataloader)

    def test_dataloader(self) -> None:
        """Test dataloader on nuPlan DB."""
        log_names = [
            '2021.07.16.20.45.29_veh-35_01095_01486',  # train
            '2021.08.17.18.54.02_veh-45_00665_01065',  # train
            '2021.06.08.12.54.54_veh-26_04262_04732',  # val
            '2021.10.06.07.26.10_veh-52_00006_00398',  # test
        ]
        overrides = [
            'scenario_builder=nuplan_mini',
            'worker=sequential',
            'splitter=nuplan',
            f'scenario_filter.log_names={log_names}',
            f'group={self.group.name}',
            f'cache.cache_path={self.cache_path}',
            'output_dir=${group}/${experiment}',
            'scenario_type_weights=default_scenario_type_weights',
        ]
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME, overrides=[*overrides, '+training=training_raster_model'])
            self.validate_cfg(cfg)
            self._run_dataloader(cfg)


if __name__ == '__main__':
    unittest.main()
