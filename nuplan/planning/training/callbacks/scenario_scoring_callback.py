import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pytorch_lightning as pl
import torch
import torch.utils.data

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.training.callbacks.utils.scene_converter import SceneConverter
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType, move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate


def _dump_scenes(scenes: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Dump a single scene file
    :param scenes: list of scenes to be written
    :param output_dir: final output directory
    """
    for scene in scenes:
        file_name = output_dir / str(scene["ego"]["timestamp_us"])

        with open(str(file_name.with_suffix('.json')), 'w') as outfile:
            json.dump(scene, outfile, indent=4)


def _score_model(
    pl_module: pl.LightningModule, features: FeaturesType, targets: TargetsType
) -> Tuple[float, FeaturesType]:
    """
    Make an inference of the input batch feature given a model and score them through their objective
    :param pl_module: lightning model
    :param features: model inputs
    :param targets: training targets
    :return: tuple of score and prediction
    """
    objectives = pl_module.objectives

    with torch.no_grad():
        pl_module.eval()
        predictions = pl_module(features)
        pl_module.train()

    score = 0.0
    for objective in objectives:
        score += objective.compute(predictions, targets).to('cpu')

    return score / len(objectives), move_features_type_to_device(predictions, torch.device('cpu'))


def _eval_model_and_write_to_scene(
    dataloader: torch.utils.data.DataLoader,
    pl_module: pl.LightningModule,
    scene_converter: SceneConverter,
    num_store: int,
    output_dir: Path,
) -> None:
    """
    Evaluate prediction of the model and write scenes based on their scores
    :param dataloader: pytorch data loader
    :param pl_module: lightning module
    :param scene_converter: converts data from the scored scenario into scene dictionary
    :param num_store: n number of scenarios to be written into scenes for each best, worst and random cases
    :param output_dir: output directory of scene file
    """
    scenario_dataset = dataloader.dataset
    score_record = torch.empty(len(scenario_dataset))

    predictions: List[TargetsType] = []

    # Obtain scores for each sample of the dataset
    for sample_idx, sample in enumerate(dataloader):
        features = cast(FeaturesType, sample[0])
        targets = cast(TargetsType, sample[1])

        score, prediction = _score_model(
            pl_module,
            move_features_type_to_device(features, pl_module.device),
            move_features_type_to_device(targets, pl_module.device),
        )

        predictions.append(prediction)
        score_record[sample_idx] = score

    # Classify score results with lower scores as best
    best_n_idx = torch.topk(score_record, num_store, largest=False).indices.tolist()
    worst_n_idx = torch.topk(score_record, num_store).indices.tolist()
    random_n_idx = random.sample(range(len(scenario_dataset)), num_store)

    # collect data to write
    for data_idx, score_type in zip((best_n_idx, worst_n_idx, random_n_idx), ('best', 'worst', 'random')):
        for idx in data_idx:
            features, targets, _ = scenario_dataset[idx]
            scenario = scenario_dataset._scenarios[idx]

            # convert data to scenes
            scenes = scene_converter(scenario, features, targets, predictions[idx])

            file_dir = output_dir / score_type / scenario.token
            if not is_s3_path(file_dir):
                file_dir.mkdir(parents=True, exist_ok=True)

            # dump scenes
            _dump_scenes(scenes, file_dir)


class ScenarioScoringCallback(pl.Callback):
    """
    Callback that performs an evaluation to score the model on each validation data.
    The n-best, n-worst and n-random data is written into a scene.

    The directory structure for the output of the scenes is:
        <output_dir>
            └── scenes
                ├── best
                │     ├── scenario_token_01
                │     │         ├── timestamp_01.json
                │     │         └── timestamp_02.json
                │     :                    :
                │     └── scenario_token_n
                ├── worst
                └── random
    """

    def __init__(self, scene_converter: SceneConverter, num_store: int, frequency: int, output_dir: Union[str, Path]):
        """
        Initialize the callback.
        :param scene_converter: Converts data from the scored scenario into scene dictionary.
        :param num_store: N number of scenarios to be written into scenes for each best, worst and random cases.
        :param frequency: Interval between epochs at which to perform the evaluation. Set 0 to skip the callback.
        :param output_dir: Output directory of scene file.
        """
        super().__init__()

        self._num_store = num_store
        self._frequency = frequency
        self._scene_converter = scene_converter
        self._output_dir = Path(output_dir) / 'scenes'

        self._val_dataloader: Optional[torch.utils.data.DataLoader] = None

    def _initialize_dataloaders(self, datamodule: pl.LightningDataModule) -> None:
        """
        Initialize the dataloaders. This makes sure that the same examples are sampled every time.
        :param datamodule: Lightning datamodule.
        """
        val_set = datamodule.val_dataloader().dataset

        assert isinstance(val_set, ScenarioDataset), "invalid dataset type, dataset must be a scenario dataset"

        self._val_dataloader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=1, shuffle=False, collate_fn=FeatureCollate()
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of each epoch validation.
        :param trainer: Lightning trainer.
        :param pl_module: lightning model.
        """
        # skip callback
        if self._frequency == 0:
            return

        assert hasattr(trainer, 'datamodule'), "Trainer missing datamodule attribute"
        assert hasattr(trainer, 'current_epoch'), "Trainer missing current_epoch attribute"

        epoch = trainer.current_epoch

        if epoch % self._frequency == 0:
            if self._val_dataloader is None:
                self._initialize_dataloaders(trainer.datamodule)

            output_dir = self._output_dir / f'epoch={epoch}'

            _eval_model_and_write_to_scene(
                self._val_dataloader, pl_module, self._scene_converter, self._num_store, output_dir
            )
