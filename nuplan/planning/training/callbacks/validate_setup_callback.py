import logging

import pytorch_lightning as pl

from nuplan.planning.training.data_loader.datamodule import DataModule
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper

logger = logging.getLogger(__name__)


class ValidateSetupCallback(pl.Callback):
    """Validate that all features are computed that are requested by a model."""

    def on_before_accelerator_backend_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called before the accelerator is being setup.
        Validate that datamodule and model do compute the same set of features, otherwise assert.
        :param trainer: Lightning trainer.
        :param pl_module: lightning model.
        """
        assert isinstance(pl_module, LightningModuleWrapper), "pl_module is not LightningModuleWrapper!"
        assert hasattr(trainer, 'datamodule'), "Trainer has no datamodule!"

        datamodule = trainer.datamodule
        assert isinstance(datamodule, DataModule), "datamodule is not DataModule!"

        # Validate input model features and builder
        model_in_feature = pl_module.model.get_list_of_required_feature()
        datamodule_feature_types = datamodule.feature_and_targets_builder.get_list_of_feature_types()

        if len(model_in_feature) != len(datamodule_feature_types):
            logger.error("Length of model input feature and builder features is not the same!")
            for model_feature in pl_module.model.get_list_of_required_feature():
                logger.error(f"Model features: {model_feature}")
            for model_feature in datamodule.feature_and_targets_builder.get_list_of_feature_types():
                logger.error(f"Datamodel features: {model_feature}")

            raise RuntimeError("Not valid model setup!")

        for feature in model_in_feature:
            assert feature in datamodule_feature_types, f"Model input feature {feature} is not in builder!"

        # Validate output model feature and builder
        model_out_feature = pl_module.model.get_list_of_computed_target()
        datamodule_target_types = datamodule.feature_and_targets_builder.get_list_of_target_types()
        assert len(model_out_feature) == len(
            datamodule_target_types
        ), "Length of model output feature and builder features is not the same!"

        for feature in model_out_feature:
            assert feature in datamodule_target_types, f"Model input feature {feature} is not in builder!"
