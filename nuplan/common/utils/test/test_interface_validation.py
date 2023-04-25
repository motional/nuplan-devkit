import abc
import unittest

from nuplan.common.utils.test_utils.interface_validation import assert_class_properly_implements_interface


class ValidationInterface:
    """
    A dummy interface class to use for testing.
    """

    def base_method(self, x: int) -> int:
        """
        Some method derived classes don't need to implement.
        :param x: The input.
        :return: The output.
        """
        return 1

    @abc.abstractmethod
    def implement_me(self, y: int) -> float:
        """
        Some method derived classes need to implement.
        :param y: The input.
        :return: The output.
        """
        raise NotImplementedError()

    def _private_method(self, a: float) -> float:
        """
        A private method.
        :param a: The input.
        :return: The output.
        """
        return a + 1.0


class SecondValidationInterface:
    """
    Another dummy interface class to use for testing
    """

    @abc.abstractmethod
    def implement_me_2(self, q: float) -> str:
        """
        A method the derived class needs to implement.
        :param q: The input.
        :return: The output.
        """
        raise NotImplementedError()


class CorrectConcrete(ValidationInterface):
    """
    A class that correctly implements the interface.
    """

    def implement_me(self, y: int) -> float:
        """
        Implemented. See interface.
        """
        return float(y + 1.0)

    def some_other_public_method(self, z: int) -> int:
        """
        An additional public method.
        :param z: The input.
        :return: The output.
        """
        return z + 1

    def _some_private_method(self, b: str) -> str:
        """
        A private method.
        :param b: The input.
        :return: The output
        """
        return b + "_foo"


class CorrectConcreteMulti(ValidationInterface, SecondValidationInterface):
    """
    A class that implements both interfaces.
    """

    def implement_me(self, y: int) -> float:
        """
        Implemented. See interface.
        """
        return float(y) + 5.0

    def implement_me_2(self, q: float) -> str:
        """
        Implemented. See interface.
        """
        return str(q)


class IncorrectConcrete(ValidationInterface):
    """
    A class that incorrectly implements the interface.
    """

    def implement_me(self, y: int) -> int:
        """
        Implemented incorrectly with wrong return type. See interface.
        """
        return int(y + 1.0)


class ConcreteMissingInterfaceMethod(ValidationInterface):
    """
    A class that is missing the interface method.
    """

    pass


class ConcreteDoesNotDerive:
    """
    A class that does not derive from the interface.
    """

    def implement_me(self, y: int) -> int:
        """
        A match for the interface, but does not properly derive.
        """
        return y + 1


class TestInterfaceValidation(unittest.TestCase):
    """
    Tests that the interface_validation utils works properly.
    """

    def test_assert_class_properly_implements_interface_correct(self) -> None:
        """
        Tests that the validation passes when a class properly implements an interface.
        """
        assert_class_properly_implements_interface(ValidationInterface, CorrectConcrete)

    def test_assert_class_properly_implements_interface_swapped_args(self) -> None:
        """
        Tests that the validation fails if the args are swapped.
        """
        with self.assertRaisesRegex(TypeError, "is not a subclass"):
            assert_class_properly_implements_interface(CorrectConcrete, ValidationInterface)

    def test_assert_class_properly_implements_interface_incorrect_method(self) -> None:
        """
        Tests that the validation fails when a class improperly implements an interface method.
        """
        with self.assertRaisesRegex(TypeError, "Types in function signature.*do not match"):
            assert_class_properly_implements_interface(ValidationInterface, IncorrectConcrete)

    def test_assert_class_properly_implements_interface_missing_method(self) -> None:
        """
        Tests that the validation fails when a class missing the interface method is passed.
        """
        with self.assertRaisesRegex(TypeError, "methods.*missing"):
            assert_class_properly_implements_interface(ValidationInterface, ConcreteMissingInterfaceMethod)

    def test_assert_class_properly_implements_interface_no_hierarchy(self) -> None:
        """
        Tests that the validation fails when the concrete does not derive from the interface.
        """
        with self.assertRaisesRegex(TypeError, "is not a subclass"):
            assert_class_properly_implements_interface(ValidationInterface, ConcreteDoesNotDerive)

    def test_assert_class_properly_implements_interface_multiple_inheritance(self) -> None:
        """
        Tests that the validation passes with the multiple inheritance use case.
        """
        assert_class_properly_implements_interface(ValidationInterface, CorrectConcreteMulti)
        assert_class_properly_implements_interface(SecondValidationInterface, CorrectConcreteMulti)


if __name__ == "__main__":
    unittest.main()
