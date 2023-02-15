import os
import unittest

from hydra import compose, initialize_config_dir

from nuplan.planning.script.run_training import CONFIG_NAME, main
from nuplan.planning.script.test.skeleton_test_train import SkeletonTestTrain


class TestTrainProfiling(SkeletonTestTrain):
    """
    Test that profiling gets generated
    """

    def test_simple_vector_model_profiling(self) -> None:
        """
        Tests that profiling file for training gets generated
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'enable_profiling=True',
                    'py_func=train',
                    '+training=training_simple_vector_model',
                    'scenario_builder=nuplan_mini',
                    'scenario_filter.limit_total_scenarios=16',
                    'splitter=nuplan',
                    'lightning.trainer.params.max_epochs=1',
                ],
            )
            main(cfg)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "profiling", "training.html")))


if __name__ == '__main__':
    unittest.main()
