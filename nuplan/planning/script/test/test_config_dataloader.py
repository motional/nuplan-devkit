import os
import tempfile
import unittest

import pytorch_lightning as pl
import torch.utils.data
from hydra import compose, initialize_config_dir
from nuplan.planning.script.builders.model_builder import build_nn_model
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.training_builder import build_lightning_datamodule
from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from omegaconf import DictConfig, OmegaConf

CONFIG_NAME = 'default_training'


class TestDataLoader(unittest.TestCase):
    """
    Tests data loading functionality
    """

    def setUp(self) -> None:
        """ Setup hydra config. """

        seed = 10
        pl.seed_everything(seed, workers=True)

        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/training/')

        # Todo: Investigate pkg in hydra
        # Since we are not using the default config in this test, we need to specify the Hydra search path in the
        # compose API override, otherwise the Jenkins build fails because bazel cannot find the simulation config file.
        common_dir = "file://" + os.path.join(main_path, '..', 'config', 'common')
        experiment_dir = "file://" + os.path.join(main_path, '..', 'experiments')
        self.search_path = f'hydra.searchpath=[{common_dir}, {experiment_dir}]'

        self.group = tempfile.TemporaryDirectory()
        self.cache_dir = os.path.join(self.group.name, 'cache_dir')

    def tearDown(self) -> None:
        """ Remove temporary folder. """

        self.group.cleanup()

    @staticmethod
    def validate_cfg(cfg: DictConfig) -> None:
        """ validate hydra config. """

        update_config_for_training(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.scenario_filter.max_scenarios_per_log = 1
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
        Iterate NUM_BATCHES of the dataloader
        :param dataloader: Data loader.
        """

        dataloader_iter = iter(dataloader)
        num_batches = 5
        iterations = min(len(dataloader), num_batches)

        for _ in range(iterations):
            next(dataloader_iter)

    def _run_dataloader(self, cfg: DictConfig) -> None:
        """
        Tests that the training dataloader can be iterated without errors.
        :param cfg: Hydra config.
        """

        worker = build_worker(cfg)
        scenario_builder = build_scenario_builder(cfg)
        planning_module = build_nn_model(cfg.model)
        datamodule = build_lightning_datamodule(cfg, scenario_builder, worker, planning_module)
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
        """ Test dataloader on nuPlan DB. """

        log_names = ['2021.05.26.20.05.14_38_1622073985538950.8_1622074969538793.5',  # train
                     '2021.07.21.02.32.00_26_1626834838399916.8_1626835894396760.2',  # train
                     '2021.06.04.19.10.47_47_1622848319071793.5_1622849413071686.2',  # val
                     '2021.05.28.21.56.29_24_1622239057169313.0_1622240664170207.2']  # test
        overrides = [
            "scenario_builder=nuplan_mini",
            "splitter=nuplan",
            "scenario_builder.nuplan.scenario_filter.log_labels=null",
            f"scenario_builder.nuplan.scenario_filter.log_names={log_names}",
            f"group={self.group.name}",
            f"cache_dir={self.cache_dir}",
        ]
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(config_name=CONFIG_NAME,
                          overrides=[self.search_path, *overrides, '+training=training_raster_model'])
            self.validate_cfg(cfg)
            self._run_dataloader(cfg)


if __name__ == '__main__':
    unittest.main()
