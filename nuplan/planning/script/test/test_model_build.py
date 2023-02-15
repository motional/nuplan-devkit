import os
import pathlib
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

CONFIG_PATH = '../config/training/'
CONFIG_NAME = 'default_training'


class TestModelBuild(unittest.TestCase):
    """Test building model."""

    def setUp(self) -> None:
        """Setup hydra config."""
        main_path = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(main_path, '../config/training/')

        self.group = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.group.name, 'cache_path')

        # ../config/common/model
        model_path = pathlib.Path(__file__).parent.parent / 'config' / 'common' / 'model'
        self.model_cfg = []
        for model_module in model_path.iterdir():
            model_name = model_module.stem
            with initialize_config_dir(config_dir=self.config_path):
                cfg = compose(
                    config_name=CONFIG_NAME,
                    overrides=[
                        '+training=training_raster_model',
                        f'model={model_name}',
                        f'group={self.group.name}',
                        f'cache.cache_path={self.cache_path}',
                    ],
                )

                self.model_cfg.append(cfg)

    def tearDown(self) -> None:
        """Remove temporary folder."""
        self.group.cleanup()

    def validate_cfg(self, cfg: DictConfig) -> None:
        """
        Validate that a model can be constructed
        :param cfg: config for model which should be constructed
        """
        lightning_module_wrapper = build_torch_module_wrapper(cfg.model)
        self.assertIsInstance(lightning_module_wrapper, TorchModuleWrapper)

        for builder in lightning_module_wrapper.get_list_of_required_feature():
            self.assertIsInstance(builder, AbstractFeatureBuilder)
        for builder in lightning_module_wrapper.get_list_of_computed_target():
            self.assertIsInstance(builder, AbstractTargetBuilder)

    def test_all_common_models(self) -> None:
        """
        Test construction of all available common models
        """
        for cfg in self.model_cfg:
            self.validate_cfg(cfg)


if __name__ == '__main__':
    unittest.main()
