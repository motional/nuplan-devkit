from typing import Any, Callable, Dict, Type, Union, cast

from hydra._internal.utils import _locate
from omegaconf import DictConfig


def is_TorchModuleWrapper_config(cfg: DictConfig) -> bool:
    """
    Check whether the config is meant for a TorchModuleWrapper
    :param cfg: config
    :return: True if model_config and checkpoint_path is in the cfg, False otherwise
    """
    return "model_config" in cfg and "checkpoint_path" in cfg


def is_target_type(cfg: DictConfig, target_type: Union[Type[Any], Callable[..., Any]]) -> bool:
    """
    Check whether the config's resolved type matches the target type or callable.
    :param cfg: config
    :param target_type: Type or callable to check against.
    :return: Whether cfg._target_ matches the target_type.
    """
    return bool(_locate(cfg._target_) == target_type)


def validate_type(instantiated_class: Any, desired_type: Type[Any]) -> None:
    """
    Validate that constructed type is indeed the desired one
    :param instantiated_class: class that was created
    :param desired_type: type that the created class should have
    """
    assert isinstance(
        instantiated_class, desired_type
    ), f"Class to be of type {desired_type}, but is {type(instantiated_class)}!"


def are_the_same_type(lhs: Any, rhs: Any) -> None:
    """
    Validate that lhs and rhs are of the same type
    :param lhs: left argument
    :param rhs: right argument
    """
    lhs_type = type(lhs)
    rhs_type = type(rhs)
    assert lhs_type == rhs_type, f"Lhs and Rhs are not of the same type! {lhs_type} != {rhs_type}!"


def validate_dict_type(instantiated_dict: Dict[str, Any], desired_type: Type[Any]) -> None:
    """
    Validate that all entries in dict is indeed the desired one
    :param instantiated_dict: dictionary that was created
    :param desired_type: type that the created class should have
    """
    for value in instantiated_dict.values():
        if isinstance(value, dict):
            validate_dict_type(value, desired_type)
        else:
            validate_type(value, desired_type)


def find_builder_in_config(cfg: DictConfig, desired_type: Type[Any]) -> DictConfig:
    """
    Find the corresponding config for the desired builder
    :param cfg: config structured as a dictionary
    :param desired_type: desired builder type
    :return: found config
    @raise ValueError if the config cannot be found for the builder
    """
    for cfg_builder in cfg.values():
        if is_target_type(cfg_builder, desired_type):
            return cast(DictConfig, cfg_builder)

    raise ValueError(f"Config does not exist for builder type: {desired_type}!")
