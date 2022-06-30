import math

import torch


def _torch_savgol_filter(
    y: torch.Tensor, window_length: int, poly_order: int, deriv_order: int, delta: float
) -> torch.Tensor:
    """
    Perform Savinsky Golay filtering on the given tensor.
    This is adapted from the scipy method `scipy.signal.savgol_filter`
        However, it currently only works with window_length of 3.
    :param y: The tensor to filter. Should be of dimension 2.
    :param window_length: The window length to use.
        Currently provided as a parameter, but for now must be 3.
    :param poly_order: The polynomial order to use.
    :param deriv_order: The order of derivitave to use.
    :coefficients: The Savinsky Golay coefficients to use.
    :return: The filtered tensor.
    """
    # TODO: port np.polyfit and remove this restriction
    if window_length != 3:
        raise ValueError("This method has unexpected edge behavior for window_length != 3.")

    if len(y.shape) != 2:
        raise ValueError(f"Unexpected input tensor shape to _torch_savgol_filter(): {y.shape}")

    # Compute the coefficients in conv format
    halflen, rem = divmod(window_length, 2)
    if rem == 0:
        pos = halflen - 0.5
    else:
        pos = float(halflen)

    # For convolution in scipy, there is a horizontal flip of x
    # But, the weight ordering between torch.nn.functional.conv1d and
    #   scipy.ndimage.convolv1d is also flipped.
    #
    # (that is, given an input tensor x and a conv filter of [1, 0, -1],
    #   the output of scipy.ndimage.conv1d will be [a, b, c, d, ...]
    #   and the output of torch.nn.functional.conv1d will be [-a, -b, -c, -d])
    #
    # So they cancel out.
    x = torch.arange(-pos, window_length - pos, dtype=torch.float32)
    order = torch.arange(poly_order + 1).reshape(-1, 1)

    yy = torch.zeros(poly_order + 1)
    A = x**order
    yy[deriv_order] = math.factorial(deriv_order) / (delta**deriv_order)

    coeffs, _, _, _ = torch.linalg.lstsq(A, yy)

    # Perform the filtering
    y_in = y.unsqueeze(1)
    coeffs_in = coeffs.reshape(1, 1, -1)
    result = torch.nn.functional.conv1d(y_in, coeffs_in, padding="same").reshape(y.shape)

    # Fix the edges
    # This only works for window_length == 3
    # A more general solution would require porting np.polyfit
    n = result.shape[1]
    result[:, 0] = y[:, 1] - y[:, 0]
    result[:, n - 1] = y[:, n - 1] - y[:, n - 2]

    return result


def _validate_approximate_derivatives_shapes(y: torch.Tensor, x: torch.Tensor) -> None:
    """
    Validates that the shapes for approximate_derivatives_tensor are correct.
    :param y: The Y input.
    :param x: The X input.
    """
    if len(y.shape) == 2 and len(x.shape) == 1 and y.shape[1] == x.shape[0]:
        return

    raise ValueError(
        f"Unexpected tensor shapes in approximate_derivatives_tensor: y.shape = {y.shape}, x.shape = {x.shape}"
    )


def approximate_derivatives_tensor(
    y: torch.Tensor, x: torch.Tensor, window_length: int = 5, poly_order: int = 2, deriv_order: int = 1
) -> torch.Tensor:
    """
    Given a time series [y], and [x], approximate [dy/dx].
    :param y: Input tensor to filter.
    :param x: Time dimension for tensor to filter.
    :param window_length: The size of the window to use.
    :param poly_order: The order of polymonial to use when filtering.
    :deriv_order: The order of derivitave to use when filtering.
    :return: The differentiated tensor.
    """
    _validate_approximate_derivatives_shapes(y, x)

    window_length = min(window_length, x.shape[0])

    if not (poly_order < window_length):
        raise ValueError(f'{poly_order} < {window_length} does not hold!')

    dx = torch.diff(x)
    min_increase = float(torch.min(dx).item())
    if min_increase <= 0:
        raise RuntimeError('dx is not monotonically increasing!')

    dx = dx.mean()

    derivative: torch.Tensor = _torch_savgol_filter(
        y,
        poly_order=poly_order,
        window_length=window_length,
        deriv_order=deriv_order,
        delta=dx,
    )

    return derivative
