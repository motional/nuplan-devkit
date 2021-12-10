import random
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.utils.data
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType, move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.visualization.raster_visualization import get_raster_with_trajectories_as_rgb


class RasterVisualizationCallback(pl.Callback):
    """
    Callbacks that visualizes model input raster and logs them in Tensorboard.
    """

    def __init__(
            self,
            images_per_tile: int,
            num_train_tiles: int,
            num_val_tiles: int,
            pixel_size: float,
    ):
        """
        Initializes the class.

        :param images_per_tile: number of images per tiles to visualize
        :param num_train_tiles: number of tiles from the training set
        :param num_val_tiles: number of tiles from the validation set
        :param pixel_size: [m] size of pixel in meters
        """
        super().__init__()

        self.custom_batch_size = images_per_tile
        self.num_train_images = num_train_tiles * images_per_tile
        self.num_val_images = num_val_tiles * images_per_tile
        self.pixel_size = pixel_size

        self.train_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.val_dataloader: Optional[torch.utils.data.DataLoader] = None

    def _initialize_dataloaders(self, datamodule: pl.LightningDataModule) -> None:
        """
        Initializes the dataloaders. This makes sure that the same examples are sampled
        every time for comparison during visualization.

        :param datamodule: lightning datamodule
        """
        train_set = datamodule.train_dataloader().dataset  # type: ignore
        val_set = datamodule.val_dataloader().dataset  # type: ignore

        self.train_dataloader = self._create_dataloader(train_set, self.num_train_images)
        self.val_dataloader = self._create_dataloader(val_set, self.num_val_images)

    def _create_dataloader(self, dataset: torch.utils.data.Dataset, num_samples: int) -> torch.utils.data.DataLoader:
        dataset_size = len(dataset)
        num_keep = min(dataset_size, num_samples)
        sampled_idxs = random.sample(range(dataset_size), num_keep)
        subset = torch.utils.data.Subset(dataset=dataset, indices=sampled_idxs)
        return torch.utils.data.DataLoader(dataset=subset, batch_size=self.custom_batch_size,
                                           collate_fn=FeatureCollate())

    def _log_from_dataloader(
            self,
            pl_module: pl.LightningModule,
            dataloader: torch.utils.data.DataLoader,
            loggers: List[Any],
            training_step: int,
            prefix: str,
    ) -> None:
        """
        Visualizes and logs all examples from the input dataloader.

        :param pl_module: lightning module used for inference
        :param dataloader: torch dataloader
        :param loggers: list of loggers from the trainer
        :param training_step: global step in training
        :param prefix: prefix to add to the log tag
        """
        for batch_idx, batch in enumerate(dataloader):
            features: FeaturesType = batch[0]
            targets: TargetsType = batch[1]
            predictions = self._infer_model(pl_module, move_features_type_to_device(features, pl_module.device))

            self._log_batch(loggers, features, targets, predictions, batch_idx, training_step, prefix)

    def _log_batch(
            self,
            loggers: List[Any],
            features: FeaturesType,
            targets: TargetsType,
            predictions: TargetsType,
            batch_idx: int,
            training_step: int,
            prefix: str,
    ) -> None:
        """
        Visualizes and logs a batch of data (features, targets, predictions) from the model.

        :param loggers: list of loggers from the trainer
        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :param batch_idx: index of total batches to visualize
        :param training_step: global trainign step
        :param prefix: prefix to add to the log tag
        """
        if 'trajectory' not in targets and 'trajectory' not in predictions:
            return

        if 'raster' in features:
            image_batch = self._get_raster_images_from_batch(
                features['raster'], targets['trajectory'], predictions['trajectory'])
        else:
            return

        tag = f'{prefix}_visualization_{batch_idx}'

        for logger in loggers:
            if isinstance(logger, torch.utils.tensorboard.writer.SummaryWriter):
                logger.add_images(
                    tag=tag,
                    img_tensor=torch.from_numpy(image_batch),
                    global_step=training_step,
                    dataformats='NHWC',
                )

    def _get_raster_images_from_batch(self, features: Raster, targets: Trajectory, predictions: Trajectory) \
            -> npt.NDArray[np.float32]:
        """
        Creates a list of RGB raster images from a batch of model data.

        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        images = list()

        for feature, target, prediction in zip(features.data, targets.data, predictions.data):
            raster = Raster.from_feature_tensor(feature)
            target_trajectory = Trajectory(target)
            predicted_trajectory = Trajectory(prediction)

            image = get_raster_with_trajectories_as_rgb(
                self.pixel_size,
                raster,
                target_trajectory,
                predicted_trajectory,
            )

            images.append(image)

        return np.asarray(images)

    def _infer_model(self, pl_module: pl.LightningModule, features: FeaturesType) -> TargetsType:
        """
        Makes an inference of the input batch features given a model.

        :param pl_module: lightning model
        :param features: model inputs
        :return: model predictions
        """
        with torch.no_grad():
            pl_module.eval()
            predictions = move_features_type_to_device(pl_module(features), torch.device('cpu'))
            pl_module.train()

        return predictions

    def on_train_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs training examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, 'datamodule'), "Trainer missing datamodule attribute"
        assert hasattr(trainer, 'global_step'), "Trainer missing global_step attribute"

        if self.train_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)  # type: ignore

        self._log_from_dataloader(
            pl_module,
            self.train_dataloader,
            trainer.logger.experiment,
            trainer.global_step,  # type: ignore
            'train',
        )

    def on_validation_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs validation examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, 'datamodule'), "Trainer missing datamodule attribute"
        assert hasattr(trainer, 'global_step'), "Trainer missing global_step attribute"

        if self.val_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)  # type: ignore

        self._log_from_dataloader(
            pl_module,
            self.val_dataloader,
            trainer.logger.experiment,
            trainer.global_step,  # type: ignore
            'val',
        )
