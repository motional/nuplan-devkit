import math
from typing import Optional

import torch


def _validate_state_se2_tensor_shape(tensor: torch.Tensor, expected_first_dim: Optional[int] = None) -> None:
    """
    Validates that a tensor is of the proper shape for a tensorized StateSE2.
    :param tensor: The tensor to validate.
    :param expected_first_dim: The expected first dimension. Can be one of three values:
        * 1: Tensor is expected to be of shape (3,)
        * 2: Tensor is expected to be of shape (N, 3)
        * None: Either shape is acceptable
    """
    expected_feature_dim = 3
    if len(tensor.shape) == 2 and tensor.shape[1] == expected_feature_dim:
        if expected_first_dim is None or expected_first_dim == 2:
            return
    if len(tensor.shape) == 1 and tensor.shape[0] == expected_feature_dim:
        if expected_first_dim is None or expected_first_dim == 1:
            return

    raise ValueError(f"Improper se2 tensor shape: {tensor.shape}")


def _validate_transform_matrix_shape(tensor: torch.Tensor) -> None:
    """
    Validates that a tensor has the proper shape for a 3x3 transform matrix.
    :param tensor: the tensor to validate.
    """
    if len(tensor.shape) == 2 and tensor.shape[0] == 3 and tensor.shape[1] == 3:
        return

    raise ValueError(f"Improper transform matrix shape: {tensor.shape}")


def state_se2_tensor_to_transform_matrix(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = torch.float64
) -> torch.Tensor:
    """
    Transforms a state of the form [x, y, heading] into a 3x3 transform matrix.
    :param input_data: the input data as a 3-d tensor.
    :return: The output 3x3 transformation matrix.
    """
    _validate_state_se2_tensor_shape(input_data, expected_first_dim=1)

    if precision is None:
        precision = input_data.dtype

    x: float = float(input_data[0].item())
    y: float = float(input_data[1].item())
    h: float = float(input_data[2].item())

    cosine: float = math.cos(h)
    sine: float = math.sin(h)

    return torch.tensor([[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]], dtype=precision)


def transform_matrix_to_state_se2_tensor(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Converts a 3x3 transformation matrix into a 3-d tensor of [x, y, heading].
    :param input_data: The 3x3 transformation matrix.
    :param precision: The precision with which to create the output tensor. If None, then it will be inferred from the input tensor.
    :return: The converted tensor.
    """
    _validate_transform_matrix_shape(input_data)

    if precision is None:
        precision = input_data.dtype

    return torch.tensor(
        [
            float(input_data[0, 2].item()),
            float(input_data[1, 2].item()),
            float(math.atan2(float(input_data[1, 0].item()), float(input_data[0, 0].item()))),
        ],
        dtype=precision,
    )


def global_state_se2_tensor_to_local(
    global_states: torch.Tensor, local_state: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transforms the StateSE2 in tensor from to the frame of reference in local_frame.

    :param global_quantities: A tensor of Nx3, where the columns are [x, y, heading].
    :param local_frame: A tensor of [x, y, h] of the frame to which to transform.
    :param precision: The precision with which to allocate the intermediate tensors. If None, then it will be inferred from the input precisions.
    :return: The transformed coordinates.
    """
    _validate_state_se2_tensor_shape(global_states, expected_first_dim=2)
    _validate_state_se2_tensor_shape(local_state, expected_first_dim=1)

    if precision is None:
        if global_states.dtype != local_state.dtype:
            raise ValueError("Mixed datatypes provided to coordinates_to_local_frame without precision specifier.")
        precision = global_states.dtype

    output = torch.zeros((global_states.shape[0], 3), dtype=global_states.dtype)
    transforms = torch.zeros((global_states.shape[0], 3, 3), dtype=precision)

    local_xform = state_se2_tensor_to_transform_matrix(local_state, precision=precision)
    local_xform_inv = torch.linalg.inv(local_xform)

    for i in range(global_states.shape[0]):
        transforms[i, :, :] = state_se2_tensor_to_transform_matrix(global_states[i, :].squeeze())

    transforms = torch.matmul(local_xform_inv, transforms)
    for i in range(transforms.shape[0]):
        xyh = transform_matrix_to_state_se2_tensor(transforms[i, :, :].squeeze())

        output[i, 0] = xyh[0]
        output[i, 1] = xyh[1]
        output[i, 2] = xyh[2]

    return output


def coordinates_to_local_frame(
    coords: torch.Tensor, anchor_state: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transform a set of [x, y] coordinates without heading to the the given frame.
    :param coords: <torch.Tensor: num_coords, 2> Coordinates to be transformed, in the form [x, y].
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param precision: The precision with which to allocate the intermediate tensors. If None, then it will be inferred from the input precisions.
    :return: <torch.Tensor: num_coords, 2> Transformed coordinates.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}")

    if precision is None:
        if coords.dtype != anchor_state.dtype:
            raise ValueError("Mixed datatypes provided to coordinates_to_local_frame without precision specifier.")
        precision = coords.dtype

    # torch.nn.functional.pad will crash with 0-length inputs.
    # In that case, there are no coordinates to transform.
    if coords.shape[0] == 0:
        return coords

    # Extract transform
    transform = state_se2_tensor_to_transform_matrix(anchor_state, precision=precision)
    transform = torch.linalg.inv(transform)

    # Transform the incoming coordinates to homogeneous coordinates
    #  So translation can be done with a simple matrix multiply.
    #
    # [x1, y1]  => [x1, y1, 1]
    # [x2, y2]     [x2, y2, 1]
    # ...          ...
    # [xn, yn]     [xn, yn, 1]
    coords = torch.nn.functional.pad(coords, (0, 1, 0, 0), "constant", value=1.0)

    # Perform the transformation, transposing so the shapes match
    coords = torch.matmul(transform, coords.transpose(0, 1))

    # Transform back from homogeneous coordinates to standard coordinates.
    #   Get rid of the scaling dimension and transpose so output shape matches input shape.
    result = coords.transpose(0, 1)
    result = result[:, :2]

    return result


def vector_set_coordinates_to_local_frame(
    coords: torch.Tensor,
    avails: torch.Tensor,
    anchor_state: torch.Tensor,
    output_precision: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Transform the vector set map element coordinates from global frame to ego vehicle frame, as specified by
        anchor_state.
    :param coords: Coordinates to transform. <torch.Tensor: num_elements, num_points, 2>.
    :param avails: Availabilities mask identifying real vs zero-padded data in coords.
        <torch.Tensor: num_elements, num_points>.
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param output_precision: The precision with which to allocate output tensors.
    :return: Transformed coordinates.
    :raise ValueError: If coordinates dimensions are not valid or don't match availabilities.
    """
    if len(coords.shape) != 3 or coords.shape[2] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}. Expected shape: (*, *, 2)")

    if coords.shape[:2] != avails.shape:
        raise ValueError(f"Mismatching shape between coords and availabilities: {coords.shape[:2]}, {avails.shape}")

    # Flatten coords from (num_map_elements, num_points_per_element, 2) to
    #   (num_map_elements * num_points_per_element, 2) for easier processing.
    num_map_elements, num_points_per_element, _ = coords.size()
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    # Apply transformation using adequate precision
    coords = coordinates_to_local_frame(coords.double(), anchor_state.double(), precision=torch.float64)

    # Reshape to original dimensionality
    coords = coords.reshape(num_map_elements, num_points_per_element, 2)

    # Output with specified precision
    coords = coords.to(output_precision)

    # ignore zero-padded data
    coords[~avails] = 0.0

    return coords
