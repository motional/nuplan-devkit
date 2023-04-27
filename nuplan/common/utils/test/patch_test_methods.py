def base_method(x: int) -> int:
    """
    A base method that should be patched.
    :param x: The input.
    :return: The output.
    """
    raise RuntimeError("Should be patched.")


def swappable_with_base_method(x: int) -> int:
    """
    A function that is swappable with base_method.
    This exists primarily to test the dynamic import capabilities of `patch_with_validation`.
    :param x: The input.
    :return: The output.
    """
    raise RuntimeError("Should not be actually run.")


def complex_method(x: int, y: int) -> int:
    """
    A mock complex method to use with the patch tests.
    :param x: One input parameter.
    :param y: The other input parameter.
    :return: The output.
    """
    # Some arbitrary math that involves base_method.
    xx = base_method(x)
    return xx * y
