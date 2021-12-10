from typing import Any, Callable, Dict, Type, cast

from hydra._internal.instantiate._instantiate2 import _resolve_target
from omegaconf import DictConfig


def is_target_type(cfg: DictConfig, target_type: Callable[..., Any]) -> bool:
    """
    Check whether desired constructed data type is of type
    :param cfg: config
    :param target_type: type to check against
    :return: True if cfg._target_ desired to construct is_type, False otherwise
    """
    return _resolve_target(cfg._target_) == target_type  # type: ignore


def validate_type(instantiated_class: Any, desired_type: Type[Any]) -> None:
    """
    Validate that constructed type is indeed the desired one
    :param instantiated_class: class that was created
    :param desired_type: type that the created class should have
    """
    assert isinstance(instantiated_class,
                      desired_type), f"Class to be of type {desired_type}, but is {type(instantiated_class)}!"


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
    for key, cfg_builder in cfg.items():
        if is_target_type(cfg_builder, desired_type):
            return cast(DictConfig, cfg_builder)
    raise ValueError(f"Config does not exist for builder type: {desired_type}!")
