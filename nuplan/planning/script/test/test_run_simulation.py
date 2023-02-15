import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.script.builders.planner_builder import build_planners
from nuplan.planning.script.run_simulation import CONFIG_NAME, run_simulation
from nuplan.planning.script.test.skeleton_test_simulation import SkeletonTestSimulation


class TestRunSimulation(SkeletonTestSimulation):
    """Test running main simulation."""

    def test_run_simulation(self) -> None:
        """
        Sanity test for passing planner as argument to run_simulation
        """
        with initialize_config_dir(config_dir=self.config_path):
            cfg = compose(
                config_name=CONFIG_NAME,
                overrides=[
                    *self.default_overrides,
                    'observation=box_observation',
                    'ego_controller=log_play_back_controller',
                    'experiment_name=simulation_test',
                ],
            )
            planner_cfg = cfg.planner
            planner = build_planners(planner_cfg, MockAbstractScenario())
            OmegaConf.set_struct(cfg, False)
            cfg.pop('planner')
            OmegaConf.set_struct(cfg, True)
            run_simulation(cfg, planner)


if __name__ == '__main__':
    unittest.main()
