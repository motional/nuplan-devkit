from typing import List, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType


def generic_tensor_shape(tensor: FeatureDataType) -> Tuple[int, ...]:
    """
    Returns the shape of generic tensor.

    :param tensor: Input tensor.
    :return: Tuple representing shape.
    """
    if isinstance(tensor, torch.Tensor) or isinstance(tensor, np.ndarray):
        return cast(Tuple[int, ...], tensor.shape)
    else:
        raise NotImplementedError


def generic_tensor_dim(tensor: FeatureDataType) -> int:
    """
    Returns number of dimensions in a generic tensor.

    :param tensor: Input tensor.
    :return: Number of dimensions.
    """
    if isinstance(tensor, torch.Tensor):
        return cast(int, tensor.dim())
    elif isinstance(tensor, np.ndarray):
        return tensor.ndim
    else:
        raise NotImplementedError


def generic_tensor_cos(tensor: FeatureDataType) -> FeatureDataType:
    """
    Performs cosine over generic tensor.

    :param tensor: Input tensor.
    :return: Tensor of same type and shape as input, containing cosine values.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.cos(tensor)
    elif isinstance(tensor, np.ndarray):
        return np.cos(tensor)
    else:
        raise NotImplementedError


def generic_tensor_sin(tensor: FeatureDataType) -> FeatureDataType:
    """
    Performs sine over generic tensor.

    :param tensor: Input tensor.
    :return: Tensor of same type and shape as input, containing sine values.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.sin(tensor)
    elif isinstance(tensor, np.ndarray):
        return np.sin(tensor)
    else:
        raise NotImplementedError


def generic_tensor_tan(tensor: FeatureDataType) -> FeatureDataType:
    """
    Performs tangent over generic tensor.

    :param tensor: Input tensor.
    :return: Tensor of same type and shape as input, containing tangent values.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.tan(tensor)
    elif isinstance(tensor, np.ndarray):
        return np.tan(tensor)
    else:
        raise NotImplementedError


def generic_tensor_hypot(first_tensor: FeatureDataType, second_tensor: FeatureDataType) -> FeatureDataType:
    """
    Performs hypot operation over two generic tensors.

    :param first_tensor: First input tensor.
    :param second_tensor: Second input tensor.
    :return: Type of same type and shape as inputs, containing hypot values.
    """
    if isinstance(first_tensor, torch.Tensor) and isinstance(second_tensor, torch.Tensor):
        return torch.hypot(first_tensor, second_tensor)
    elif isinstance(second_tensor, np.ndarray) and isinstance(second_tensor, np.ndarray):
        return np.hypot(first_tensor, second_tensor)
    else:
        raise NotImplementedError


def generic_tensor_stack(tensors: List[FeatureDataType], dim: int) -> FeatureDataType:
    """
    Stacks several generic tensors into a singel tensor.

    :param tensors: List of input tensors.  Must be of same type and compatible shapes.
    :param dim: Index of dimension along which to perform stacking.
    :return: Stacked tensor.
    """
    if isinstance(tensors[0], torch.Tensor):
        return torch.stack(tensors, dim=dim)
    elif isinstance(tensors[0], np.ndarray):
        return np.stack(tensors, axis=dim)
    else:
        raise NotImplementedError


def generic_tensor_like(data: npt.ArrayLike, like_tensor: FeatureDataType) -> FeatureDataType:
    """
    Constructs a new generic tensor with the same type as an existing generic tensor.

    :param data: Data to be contained in new tensor.
    :param like_tensor: Used to determine type of output tensor.
    :return: Generic tensor containing data and of same type as like_tensor.
    """
    if isinstance(like_tensor, torch.Tensor):
        return torch.as_tensor(data, dtype=like_tensor.dtype, device=like_tensor.device)
    elif isinstance(like_tensor, np.ndarray):
        return np.array(data, like=like_tensor)
    else:
        raise NotImplementedError


def generic_tensor_zeros_like(like_tensor: FeatureDataType) -> FeatureDataType:
    """
    Constructs a new generic tensor containing all zeros with the same type and shape
    as an existing generic tensor.

    :param like_tensor: Used to determine type and shape of output tensor.
    :return: Tensor containing all zeros with same shape and type as like_tensor.
    """
    if isinstance(like_tensor, torch.Tensor):
        return torch.zeros_like(like_tensor)
    elif isinstance(like_tensor, np.ndarray):
        return np.zeros_like(like_tensor)
    else:
        raise NotImplementedError


def generic_tensor_to_numpy64(tensor: FeatureDataType) -> npt.NDArray[np.float64]:
    """
    Converts generic tensor to 64-bit numpy array.

    :param tensor: Input tensor.
    :return: 64-bit numpy array with same data as input tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return cast(npt.NDArray[np.float64], tensor.detach().cpu().numpy().astype(np.float64))
    elif isinstance(tensor, np.ndarray):
        return tensor.astype(np.float64)
    else:
        raise NotImplementedError


def generic_tensor_squeeze(tensor: FeatureDataType) -> FeatureDataType:
    """
    Removes all dimensions of size 1 from a tensor.

    :param tensor: Input tensor.
    :return: Tensor of same type and data as input but with dimensions of size 1 removed.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.squeeze(tensor)
    elif isinstance(tensor, np.ndarray):
        return np.squeeze(tensor)
    else:
        raise NotImplementedError
