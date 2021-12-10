import os
import pathlib
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from nuplan.planning.script.builders.model_builder import build_nn_model
from nuplan.planning.training.modeling.nn_model import NNModule
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from omegaconf import DictConfig

CONFIG_PATH = '../config/training/'
CONFIG_NAME = 'default_training'


class TestModelBuild(unittest.TestCase):
    """ Test building model. """

    def setUp(self) -> None:
        """ Setup hydra config. """

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

        # ../config/common/model
        model_path = pathlib.Path(__file__).parent.parent / "config" / "common" / "model"
        self.model_cfg = []
        for model_module in model_path.iterdir():
            model_name = model_module.stem
            with initialize_config_dir(config_dir=self.config_path):
                cfg = compose(config_name=CONFIG_NAME,
                              overrides=[self.search_path,
                                         "+training=training_raster_model",
                                         f"model={model_name}",
                                         f"group={self.group.name}",
                                         f"cache_dir={self.cache_dir}"])

                self.model_cfg.append(cfg)

    def tearDown(self) -> None:
        """ Remove temporary folder. """

        self.group.cleanup()

    def validate_cfg(self, cfg: DictConfig) -> None:
        """
        Validate that a model can be constructed
        :param cfg: config for model which should be constructed
        """
        planning_module = build_nn_model(cfg.model)
        self.assertIsInstance(planning_module, NNModule)

        for builder in planning_module.get_list_of_required_feature():
            self.assertIsInstance(builder, AbstractFeatureBuilder)
        for builder in planning_module.get_list_of_computed_target():
            self.assertIsInstance(builder, AbstractTargetBuilder)

    def test_all_common_models(self) -> None:
        """
        Test construction of all available common models
        """

        for cfg in self.model_cfg:
            self.validate_cfg(cfg)


if __name__ == '__main__':
    unittest.main()
