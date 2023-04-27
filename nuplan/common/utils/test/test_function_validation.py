import textwrap
import unittest
from dataclasses import dataclass

# Disable import warnings on a few of these lines - only used in dynamic function creation
from typing import Dict, List, Optional, cast  # noqa: F401

import numpy as np  # noqa: F401
import numpy.typing as npt  # noqa: F401

from nuplan.common.utils.test_utils.function_validation import assert_functions_swappable


@dataclass(frozen=True)
class MethodSpecification:
    """
    A contract class for specifying a unit test method to be generated.
    """

    name: str
    input_args: Dict[str, str]
    kw_only_args: Optional[Dict[str, str]]
    return_type: str


def _get_method_text(spec: MethodSpecification) -> str:
    """
    Gets the text of a method to use for unit testing.
    This method does nothing and raises a `NotImplementedError()` if it is called.
    :param spec: The method specification.
    """
    input_signature_items = [f"{kvp[0]}: {kvp[1]}" for kvp in spec.input_args.items()]

    if spec.kw_only_args is not None:
        input_signature_items.append("*")
        input_signature_items += [f"{kvp[0]}: {kvp[1]}" for kvp in spec.kw_only_args.items()]

    input_signature = ", ".join(input_signature_items)

    method_text = textwrap.dedent(
        f"""
        def {spec.name}({input_signature}) -> {spec.return_type}:
            raise NotImplementedError()
        """
    )

    return method_text


class TestFunctionValidation(unittest.TestCase):
    """
    A class to test that the function validation method works properly.
    """

    def test_assert_functions_swappable(self) -> None:
        """
        Tests that the assert_functions_swappable method functions properly.
        """
        # Naming convention:
        # * Args separated by underscore, with all but last being passed in args.
        # * If default value is provided, then "d" is added as a suffix.
        # * If arg is keyword only, then "k" is added as a suffix.
        #
        # Some functions have the "_duplicate_of" prefix, which should be swappable with functions without that prefix.
        #  e.g. '_duplicate_of_int_int' should be swappable with '_int_int', but not '_int_float'
        test_methods: List[MethodSpecification] = [
            MethodSpecification(
                name="_none_none",
                input_args={},
                kw_only_args=None,
                return_type="None",
            ),
            MethodSpecification(
                name="_int_none",
                input_args={"x": "int"},
                kw_only_args=None,
                return_type="None",
            ),
            MethodSpecification(
                name="_intd1_none",
                input_args={"x": "int = 1"},
                kw_only_args=None,
                return_type="None",
            ),
            MethodSpecification(
                name="_intk1_none",
                input_args={},
                kw_only_args={"x": "int = 1"},
                return_type="None",
            ),
            MethodSpecification(
                name="_int_int",
                input_args={"x": "int"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_intd1_int",
                input_args={"x": "int = 1"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_intd2_int",
                input_args={"x": "int = 2"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_int_float",
                input_args={"x": "int"},
                kw_only_args=None,
                return_type="float",
            ),
            MethodSpecification(
                name="_float_int",
                input_args={"x": "float"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_list_int_int",
                input_args={"x": "List[int]"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_list_float_int",
                input_args={"x": "List[float]"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_int_int_int",
                input_args={"x": "int", "y": "int"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_int_intk1_int",
                input_args={"x": "int"},
                kw_only_args={"y": "int = 1"},
                return_type="int",
            ),
            MethodSpecification(
                name="_int_intk2_int",
                input_args={"x": "int"},
                kw_only_args={"y": "int = 2"},
                return_type="int",
            ),
            MethodSpecification(
                name="_float_int_int",
                input_args={"x": "float", "y": "int"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_ndarray_float32_int",
                input_args={"x": "npt.NDArray[np.float32]"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_ndarray_float64_int",
                input_args={"x": "npt.NDArray[np.float64]"},
                kw_only_args=None,
                return_type="int",
            ),
            MethodSpecification(
                name="_duplicate_of_int_int",
                input_args={"x": "int"},
                kw_only_args=None,
                return_type="int",
            ),
        ]

        for spec in test_methods:
            method_text = _get_method_text(spec)
            exec(method_text)

        for first_func_definition in test_methods:
            for second_func_definition in test_methods:
                first_func_name = first_func_definition.name
                second_func_name = second_func_definition.name

                first_func = locals()[first_func_name]
                second_func = locals()[second_func_name]

                if first_func_name.replace("_duplicate_of", "") != second_func_name.replace("_duplicate_of", ""):
                    with self.assertRaises(TypeError):
                        assert_functions_swappable(first_func, second_func)
                else:
                    assert_functions_swappable(first_func, second_func)


if __name__ == "__main__":
    unittest.main()
