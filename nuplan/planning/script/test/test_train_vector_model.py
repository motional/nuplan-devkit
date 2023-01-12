import unittest

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain


class TestTrainVectorModel(SkeletonTestTrain):
    """
    Test experiments: simple_vector_model, vector_model
    """

    def test_open_loop_training_simple_vector_model(self) -> None:
        """
        Tests simple vector model training in open loop.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'py_func=train',
                    '+training=training_simple_vector_model',
                    'scenario_builder=nuplan_mini',
                    'scenario_filter.limit_total_scenarios=16',
                    'splitter=nuplan',
                    'lightning.trainer.params.max_epochs=1',
                ],
            )
            main(cfg)

    def test_open_loop_training_vector_model(self) -> None:
        """
        Tests vector model training in open loop.
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    self.search_path,
                    *self.default_overrides,
                    'py_func=train',
                    '+training=training_vector_model',
                    'scenario_builder=nuplan_mini',
                    'scenario_filter.limit_total_scenarios=16',
                    'splitter=nuplan',
                    'model.num_res_blocks=1',
                    'model.num_attention_layers=1',
                    'model.feature_dim=8',
                    'lightning.trainer.params.max_epochs=1',
                ],
            )
            main(cfg)


if __name__ == '__main__':
    unittest.main()
