import textwrap
from typing import Any, Callable


def _assert_function_signature_types_match(
    first_func: Callable[..., Any],
    second_func: Callable[..., Any],
) -> None:
    """
    Checks that the types in two method's function signatures match.
    If a difference is found, a TypeError is raised.
    :param first_func: The first function that is being seconded.
    :param second_func: The second that is being used.
    """
    first_annotations = first_func.__annotations__
    second_annotations = second_func.__annotations__

    if first_annotations != second_annotations:
        first_annotations_values = list(first_annotations.items()) if first_annotations is not None else []
        second_annotations_values = list(second_annotations.items()) if second_annotations is not None else []

        first_annotations_str = ", ".join([f"{kvp[0]}: {kvp[1]}" for kvp in first_annotations_values])
        second_annotations_str = ", ".join([f"{kvp[0]}: {kvp[1]}" for kvp in second_annotations_values])

        raise TypeError(
            textwrap.dedent(
                f"""
                Types in function signature for {first_func} do not match.
                First func: {first_annotations_str}
                Second func: {second_annotations_str}
            """
            )
        )


def _assert_function_defaults_match(
    first_func: Callable[..., Any],
    second_func: Callable[..., Any],
) -> None:
    """
    Checks that the defaults set for the functions match.
    If a difference is found, a TypeError is raised.
    :param first_func: The first function that is being seconded.
    :param second_func: The second that is being used.
    """
    first_defaults = first_func.__defaults__
    second_defaults = second_func.__defaults__

    if first_defaults != second_defaults:
        raise TypeError(
            textwrap.dedent(
                f"""
                Default values for function {first_func} do not match.
                First func: {first_defaults}
                Second func: {second_defaults}
            """
            )
        )


def _assert_function_kwdefaults_match(
    first_func: Callable[..., Any],
    second_func: Callable[..., Any],
) -> None:
    """
    Checks that the kwdefaults set for the functions match.
    If a difference is found, a TypeError is raised.
    :param first_func: The first function that is being seconded.
    :param second_func: The second that is being used.
    """
    first_kwdefaults = first_func.__kwdefaults__
    second_kwdefaults = second_func.__kwdefaults__

    if first_kwdefaults != second_kwdefaults:
        first_kwdefault_values = list(first_kwdefaults.items()) if first_kwdefaults is not None else []
        second_kwdefault_values = list(second_kwdefaults.items()) if second_kwdefaults is not None else []

        first_kwdefault_str = ", ".join([f"{kvp[0]}: {kvp[1]}" for kvp in first_kwdefault_values])
        second_kwdefault_str = ", ".join([f"{kvp[0]}: {kvp[1]}" for kvp in second_kwdefault_values])

        raise TypeError(
            textwrap.dedent(
                f"""
                Kwdefaults values in function signature for {first_func} do not match.
                First func: {first_kwdefault_str}
                Second func: {second_kwdefault_str}
            """
            )
        )


def assert_functions_swappable(first_func: Callable[..., Any], second_func: Callable[..., Any]) -> None:
    """
    Asserts that a second function is swappable for the supplied first function.
    "Swappable" means that they contain the same arguments, same default arguments, and same return type.
    :param first_func: The first func that is being replaced.
    :param second_func: The second func that is being replaced.
    """
    _assert_function_signature_types_match(first_func, second_func)
    _assert_function_defaults_match(first_func, second_func)
    _assert_function_kwdefaults_match(first_func, second_func)

    # TODO: any other heuristics? Maybe examine __code__ or __closure__?
