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


def _validate_state_se2_tensor_batch_shape(tensor: torch.Tensor) -> None:
    """
    Validates that a tensor is of the proper shape for a batch of tensorized StateSE2.
    :param tensor: The tensor to validate.
    """
    expected_feature_dim = 3
    if len(tensor.shape) == 2 and tensor.shape[1] == expected_feature_dim:
        return

    raise ValueError(f"Improper se2 tensor batch shape: {tensor.shape}")


def _validate_transform_matrix_shape(tensor: torch.Tensor) -> None:
    """
    Validates that a tensor has the proper shape for a 3x3 transform matrix.
    :param tensor: the tensor to validate.
    """
    if len(tensor.shape) == 2 and tensor.shape[0] == 3 and tensor.shape[1] == 3:
        return

    raise ValueError(f"Improper transform matrix shape: {tensor.shape}")


def _validate_transform_matrix_batch_shape(tensor: torch.Tensor) -> None:
    """
    Validates that a tensor has the proper shape for a 3x3 transform matrix.
    :param tensor: the tensor to validate.
    """
    if len(tensor.shape) == 3 and tensor.shape[1] == 3 and tensor.shape[2] == 3:
        return

    raise ValueError(f"Improper transform matrix shape: {tensor.shape}")


def state_se2_tensor_to_transform_matrix(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
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

    return torch.tensor(
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]], dtype=precision, device=input_data.device
    )


def state_se2_tensor_to_transform_matrix_batch(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transforms a tensor of states of the form Nx3 (x, y, heading) into a Nx3x3 transform tensor.
    :param input_data: the input data as a Nx3 tensor.
    :param precision: The precision with which to create the output tensor. If None, then it will be inferred from the input tensor.
    :return: The output Nx3x3 batch transformation tensor.
    """
    _validate_state_se2_tensor_batch_shape(input_data)

    if precision is None:
        precision = input_data.dtype

    # Transform the incoming coordinates so transformation can be done with a simple matrix multiply.
    #
    # [x1, y1, phi1]  => [x1, y1, cos1, sin1, 1]
    # [x2, y2, phi2]     [x2, y2, cos2, sin2, 1]
    # ...          ...
    # [xn, yn, phiN]     [xn, yn, cosN, sinN, 1]
    processed_input = torch.column_stack(
        (
            input_data[:, 0],
            input_data[:, 1],
            torch.cos(input_data[:, 2]),
            torch.sin(input_data[:, 2]),
            torch.ones_like(input_data[:, 0], dtype=precision),
        )
    )

    # See below for reshaping example
    reshaping_tensor = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=precision,
        device=input_data.device,
    )
    # Builds the transform matrix
    # First computes the components of each transform as rows of a Nx9 tensor, and then reshapes to a Nx3x3 tensor
    # Below is outlined how the Nx9 representation looks like (s1 and c1 are cos1 and sin1)
    # [x1, y1, c1, s1, 1]  => [c1, -s1, x1, s1, c1, y1, 0, 0, 1]  =>  [[c1, -s1, x1], [s1, c1, y1], [0, 0, 1]]
    # [x2, y2, c2, s2, 1]     [c2, -s2, x2, s2, c2, y2, 0, 0, 1]  =>  [[c2, -s2, x2], [s2, c2, y2], [0, 0, 1]]
    # ...          ...
    # [xn, yn, cN, sN, 1]     [cN, -sN, xN, sN, cN, yN, 0, 0, 1]
    return (processed_input @ reshaping_tensor).reshape(-1, 3, 3)


def transform_matrix_to_state_se2_tensor(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Converts a Nx3x3 transformation tensor into a Nx3 tensor of [x, y, heading] rows.
    :param input_data: The Nx3x3 transformation matrix.
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


def transform_matrix_to_state_se2_tensor_batch(input_data: torch.Tensor) -> torch.Tensor:
    """
    Converts a Nx3x3 batch transformation matrix into a Nx3 tensor of [x, y, heading] rows.
    :param input_data: The 3x3 transformation matrix.
    :return: The converted tensor.
    """
    _validate_transform_matrix_batch_shape(input_data)

    # Picks the entries, the third column will be overwritten with the headings [x, y, _]
    first_columns = input_data[:, :, 0].reshape(-1, 3)
    angles = torch.atan2(first_columns[:, 1], first_columns[:, 0])

    result = input_data[:, :, 2]
    result[:, 2] = angles

    return result


def global_state_se2_tensor_to_local(
    global_states: torch.Tensor, local_state: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transforms the StateSE2 in tensor from to the frame of reference in local_frame.

    :param global_states: A tensor of Nx3, where the columns are [x, y, heading].
    :param local_state: A tensor of [x, y, h] of the frame to which to transform.
    :param precision: The precision with which to allocate the intermediate tensors. If None, then it will be inferred from the input precisions.
    :return: The transformed coordinates.
    """
    _validate_state_se2_tensor_shape(global_states, expected_first_dim=2)
    _validate_state_se2_tensor_shape(local_state, expected_first_dim=1)

    if precision is None:
        if global_states.dtype != local_state.dtype:
            raise ValueError("Mixed datatypes provided to coordinates_to_local_frame without precision specifier.")
        precision = global_states.dtype

    local_xform = state_se2_tensor_to_transform_matrix(local_state, precision=precision)
    local_xform_inv = torch.linalg.inv(local_xform)

    transforms = state_se2_tensor_to_transform_matrix_batch(global_states, precision=precision)

    transforms = torch.matmul(local_xform_inv, transforms)

    output = transform_matrix_to_state_se2_tensor_batch(transforms)

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
