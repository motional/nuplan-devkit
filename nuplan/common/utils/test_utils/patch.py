import contextlib
import importlib
import unittest.mock
from typing import Any, Callable, Generator, Optional, cast

from nuplan.common.utils.test_utils.function_validation import assert_functions_swappable


def _get_method_from_import(import_str: str) -> Callable[..., Any]:
    """
    Gets the method referenced by an import in the form that `unittest.mock.patch` expects.
    That is, given the string "foo.bar.baz.qux", imports "qux" from "foo.bar.baz"
    This is not a general purpose utility, so other import mechanisms (e.g. "from x import y")
      will not work.
    :param import_str: The import str.
    :return: The method.
    """
    import_path, method_name = import_str.rsplit(".", 1)
    module = importlib.import_module(import_path)
    method = cast(Callable[..., Any], getattr(module, method_name))
    return method


@contextlib.contextmanager
def patch_with_validation(
    method_to_patch: str,
    patch_function: Callable[..., Any],
    override_function: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> Generator[Callable[..., Any], None, None]:
    """
    Wraps unittest.mock.patch, injecting the function signature validation.
    :param method_to_patch: The dot-string method to patch (e.g. "my.python.file.mymethod")
    :param patch_function: The function to use for the patch.
    :param override_function: The function to use for validation. If not provided, `method_to_patch` will be imported and used.
      The intent is to provide an escape hatch in the instance automatic lookup via dot-string for method_to_patch does not work.
    :param kwargs: The additional keyword arguments passed to unittest.mock.patch.
    """
    if override_function is None:
        override_function = _get_method_from_import(method_to_patch)

    assert_functions_swappable(override_function, patch_function)

    with unittest.mock.patch(method_to_patch, patch_function, **kwargs) as mock_obj:
        yield mock_obj
