import inspect
import textwrap
from typing import Any, Callable, Dict, Set, Type

from nuplan.common.utils.test_utils.function_validation import assert_functions_swappable


def _assert_derived_is_child_of_base(interface_class_type: Type[Any], derived_class_type: Type[Any]) -> None:
    """
    Checks that derived is an instance of base.
    Throws a TypeError if it is not.
    :param interface_class: The interface class.
    :param derived_class: The derived class.
    """
    if not issubclass(derived_class_type, interface_class_type):
        raise TypeError(
            textwrap.dedent(
                f"""
            {derived_class_type} is not a subclass of {interface_class_type}.
            """
            )
        )


def _get_public_methods(class_type: Type[Any], only_abstract: bool) -> Dict[str, Callable[..., Any]]:
    """
    Get all of the public methods exposed on a class.
    This excludes magic methods and underscore-prefixed methods.
    :param class_type: The type of the class for which to get the methods.
    :param only_abstract: If true, only returns abstract methods.
    :return: Mapping of (method_name -> function_object).
    """
    all_functions = {
        tup[0]: tup[1]
        for tup in inspect.getmembers(class_type, predicate=inspect.isfunction)
        if tup[1].__qualname__.startswith(class_type.__qualname__)
    }

    public_functions = {key: all_functions[key] for key in all_functions if not key.startswith("_")}

    # Currently, not all of our classes inherit from abc.ABCMeta.
    # So, they don't necessarily have __abstractmethods__.
    if only_abstract:
        public_functions = {
            key: public_functions[key]
            for key in public_functions
            if hasattr(public_functions[key], "__isabstractmethod__") and public_functions[key].__isabstractmethod__
        }

    return public_functions


def _assert_abstract_methods_present(
    interface_class_type: Type[Any],
    derived_class_type: Type[Any],
    interface_abstract_method_names: Set[str],
    derived_public_method_names: Set[str],
) -> None:
    """
    Asserts that all public methods in interface are in derived.
    :param interface_class_type: The class type of interface.
    :param derived_class_type: The class type of derived.
    :param interface_abstract_method_names: The interface abstract method names.
    :param derived_public_method_names: The derived public method names.
    """
    missing_methods = [im for im in interface_abstract_method_names if im not in derived_public_method_names]

    if len(missing_methods) > 0:
        missing_method_names = ", ".join(missing_methods)
        raise TypeError(
            textwrap.dedent(
                f"""
            The following methods are missing in {derived_class_type}, which are abstract in {interface_class_type}: {missing_method_names}
            """
            )
        )


def assert_class_properly_implements_interface(
    interface_class_type: Type[Any],
    derived_class_type: Type[Any],
) -> None:
    """
    Asserts that a particular class implements a specified interface.
    This is done with the following checks:
        * Makes sure that derived_class is a subclass of interface_class.
        * Checks that all abstract public methods in interface are in derived.
        * Checks that the function signatures in derived class are swappable with abstract methods in interface.
    If the checks fail, a TypeError is raised.
    :param interface_class_type: The type of the interface class.
    :param derived_class_type: The type of the derived class.
    """
    _assert_derived_is_child_of_base(interface_class_type, derived_class_type)

    interface_abstract_methods = _get_public_methods(interface_class_type, only_abstract=True)
    derived_public_methods = _get_public_methods(derived_class_type, only_abstract=False)

    _assert_abstract_methods_present(
        interface_class_type,
        derived_class_type,
        {k for k in interface_abstract_methods.keys()},
        {k for k in derived_public_methods.keys()},
    )

    for key in interface_abstract_methods:
        interface_method = interface_abstract_methods[key]
        derived_method = derived_public_methods[key]
        assert_functions_swappable(interface_method, derived_method)
