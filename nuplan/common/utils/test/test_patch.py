import unittest
import unittest.mock

from nuplan.common.utils.test.patch_test_methods import complex_method, swappable_with_base_method
from nuplan.common.utils.test_utils.patch import patch_with_validation


class TestPatch(unittest.TestCase):
    """
    A class to test the patch utils.
    """

    def test_patch_with_validation_correct_patch(self) -> None:
        """
        Tests that the patch works with a correct patch.
        """

        def correct_patch(x: int) -> int:
            """
            A correct patch for the base_method.
            :param x: The input.
            :return: The output.
            """
            return x + 1

        with patch_with_validation("nuplan.common.utils.test.patch_test_methods.base_method", correct_patch):
            result = complex_method(x=1, y=2)
            self.assertEqual(4, result)

    def test_patch_with_validation_correct_patch_direct(self) -> None:
        """
        Tests that the patch works with a correct patch that is directly provided.
        """

        def correct_patch(x: int) -> int:
            """
            A correct patch for the base_method.
            :param x: The input.
            :return: The output.
            """
            return x + 1

        with patch_with_validation(
            "nuplan.common.utils.test.patch_test_methods.base_method",
            correct_patch,
            override_function=swappable_with_base_method,
        ):
            result = complex_method(x=1, y=2)
            self.assertEqual(4, result)

    def test_patch_raises_with_incorrect_patch(self) -> None:
        """
        Tests that an incorrect patch causes an error to be rasied.
        """

        def incorrect_patch(x: float) -> int:
            """
            An incorrect patch for the _base_method.
            :param x: The input.
            :return: The output.
            """
            return int(x) + 1

        with self.assertRaises(TypeError):
            with patch_with_validation("nuplan.common.utils.test.patch_test_methods.base_method", incorrect_patch):
                _ = complex_method(x=1, y=2)

    def test_patch_raises_with_incorrect_patch_direct(self) -> None:
        """
        Tests that an incorrect patch causes an error to be rasied.
        """

        def incorrect_patch(x: float) -> int:
            """
            An incorrect patch for the _base_method.
            :param x: The input.
            :return: The output.
            """
            return int(x) + 1

        with self.assertRaises(TypeError):
            with patch_with_validation(
                "nuplan.common.utils.test.patch_test_methods.base_method",
                incorrect_patch,
                override_function=swappable_with_base_method,
            ):
                _ = complex_method(x=1, y=2)


if __name__ == "__main__":
    unittest.main()
