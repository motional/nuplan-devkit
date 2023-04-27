import json
import pathlib
import tempfile
import unittest
from typing import Tuple
from unittest.mock import Mock

import torch

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.training.callbacks.scenario_scoring_callback import ScenarioScoringCallback, _score_model
from nuplan.planning.training.callbacks.utils.scenario_scene_converter import ScenarioSceneConverter
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.test.dummy_vectormap_builder import DummyVectorMapFeature


def mock_compute_features(scenario: AbstractScenario) -> Tuple[FeaturesType, TargetsType, None]:
    """
    Mock feature computation.
    :param scenario: Input scenario to extract features from.
    :return: Extracted features and targets.
    """
    mission_goal = scenario.get_mission_goal()
    data1 = torch.tensor(mission_goal.x)
    data2 = torch.tensor(mission_goal.y)
    data3 = torch.tensor(mission_goal.heading)

    mock_feature = DummyVectorMapFeature(data1=[data1], data2=[data2], data3=[{"test": data3}])
    mock_output = {'mock_feature': mock_feature}
    mock_cache_metadata = None
    return mock_output, mock_output, mock_cache_metadata


def mock_predict(features: FeaturesType) -> FeaturesType:
    """
    Mock prediction function.
    :param features: Input feature tensor.
    :return: Predicted tensor.
    """
    return features


def mock_compute_objective(prediction: FeaturesType, target: TargetsType) -> torch.Tensor:
    """
    Mock computation of objective.
    :param prediction: Prediction tensor.
    :param target: Target tensor.
    :return: Computed objective tensor.
    """
    return target['mock_feature'].data1[0]


class TestScenarioScoringCallback(unittest.TestCase):
    """Test scenario scoring callback"""

    def setUp(self) -> None:
        """Set up test case."""
        self.output_dir = tempfile.TemporaryDirectory()

        # setup scenario dataset
        preprocessor = Mock()
        preprocessor.compute_features.side_effect = mock_compute_features

        self.mock_scenarios = [
            MockAbstractScenario(mission_goal=StateSE2(x=1.0, y=0.0, heading=0.0)),
            MockAbstractScenario(mission_goal=StateSE2(x=0.0, y=0.0, heading=0.0)),
        ]
        # Will be the same for both scenarios
        self.scenario_time_stamp = self.mock_scenarios[0]._initial_time_us

        mock_scenario_dataset = ScenarioDataset(scenarios=self.mock_scenarios, feature_preprocessor=preprocessor)

        # setup datamodule
        mock_datamodule = Mock()
        mock_datamodule.val_dataloader().dataset = mock_scenario_dataset

        # setup trainer
        self.trainer = Mock()
        self.trainer.datamodule = mock_datamodule
        self.trainer.current_epoch = 1

        # setup objective
        mock_objective = Mock()
        mock_objective.compute.side_effect = mock_compute_objective

        # setup lightning module
        self.pl_module = Mock()
        self.pl_module.device = "cpu"
        self.pl_module.side_effect = mock_predict
        self.pl_module.objectives = [mock_objective]

        # setup callback
        scenario_converter = ScenarioSceneConverter(ego_trajectory_horizon=1, ego_trajectory_poses=2)
        self.callback = ScenarioScoringCallback(
            scene_converter=scenario_converter, num_store=1, frequency=1, output_dir=self.output_dir.name
        )

        self.callback._initialize_dataloaders(self.trainer.datamodule)

    def test_initialize_dataloaders(self) -> None:
        """
        Test callback dataloader initialization.
        """
        invalid_datamodule = Mock()
        invalid_datamodule.val_dataloader().dataset = None

        # test invalid dataset assertion
        with self.assertRaises(AssertionError):
            self.callback._initialize_dataloaders(invalid_datamodule)

        # test valid dataset instance
        self.callback._initialize_dataloaders(self.trainer.datamodule)
        self.assertIsInstance(self.callback._val_dataloader, torch.utils.data.DataLoader)

    def test_score_model(self) -> None:
        """
        Test scoring of the model with mock features.
        """
        data1 = torch.tensor(1)
        data2 = torch.tensor(2)
        data3 = torch.tensor(3)

        mock_feature = DummyVectorMapFeature(data1=[data1], data2=[data2], data3=[{"test": data3}])
        mock_input = {'mock_feature': mock_feature}
        score, prediction = _score_model(self.pl_module, mock_input, mock_input)
        self.assertEqual(score, mock_feature.data1[0])
        self.assertEqual(prediction, mock_input)

    def test_on_validation_epoch_end(self) -> None:
        """
        Test on validation callback.
        """
        BEST_INDEX = 1
        WORST_INDEX = 0

        self.callback._initialize_dataloaders(self.trainer.datamodule)
        self.callback.on_validation_epoch_end(self.trainer, self.pl_module)

        # Assert files are generated
        best_score_path = pathlib.Path(
            self.output_dir.name
            + f"/scenes/epoch={self.trainer.current_epoch}"
            + f"/best/{self.mock_scenarios[BEST_INDEX].token}/{self.scenario_time_stamp.time_us}.json"
        )
        self.assertTrue(best_score_path.exists())

        worst_score_path = pathlib.Path(
            self.output_dir.name
            + f"/scenes/epoch={self.trainer.current_epoch}"
            + f"/worst/{self.mock_scenarios[WORST_INDEX].token}/{self.scenario_time_stamp.time_us}.json"
        )
        self.assertTrue(worst_score_path.exists())

        # We don't know what the random scenario index will be, so we fuzzy match scenario token
        random_score_dir = pathlib.Path(self.output_dir.name + f"/scenes/epoch={self.trainer.current_epoch}/random/")
        random_score_paths = list(random_score_dir.glob(f"*/{self.scenario_time_stamp.time_us}.json"))
        self.assertEqual(len(random_score_paths), 1)

        # Make sure the right json files are generated
        with open(str(best_score_path), "r") as f:
            best_data = json.load(f)

        with open(str(worst_score_path), "r") as f:
            worst_data = json.load(f)

        self.assertEqual(worst_data["goal"]["pose"][0], self.mock_scenarios[WORST_INDEX].get_mission_goal().x)
        self.assertEqual(best_data["goal"]["pose"][0], self.mock_scenarios[BEST_INDEX].get_mission_goal().x)


if __name__ == '__main__':
    unittest.main()
