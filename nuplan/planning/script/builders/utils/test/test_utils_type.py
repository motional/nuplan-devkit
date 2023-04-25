import unittest
from dataclasses import dataclass

import hydra.utils
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import (
    are_the_same_type,
    find_builder_in_config,
    is_target_type,
    is_TorchModuleWrapper_config,
    validate_dict_type,
    validate_type,
)


@dataclass
class TestUtilsTypeMockType:
    """Mock class for testing purposes."""

    a: int = 0  # mock member
    b: float = 0  # mock member


@dataclass
class TestUtilsTypeAnotherMockType:
    """Another mock class for testing purposes."""

    c: int = 0  # mock member


class TestUtilsType(unittest.TestCase):
    """Test utils_type functions."""

    def test_is_TorchModuleWrapper_config(self) -> None:
        """Tests that is_TorchModuleWrapper_config works as expected."""
        # Nominal cases
        mock_config = DictConfig(
            {
                "model_config": "some_value",
                "checkpoint_path": "some_value",
                "some_other_key": "some_value",
            }
        )
        expect_true = is_TorchModuleWrapper_config(mock_config)
        self.assertTrue(expect_true)

        mock_config.pop("some_other_key")  # pop a key that's not related to the function in test
        expect_true = is_TorchModuleWrapper_config(mock_config)
        self.assertTrue(expect_true)

        # Failing cases
        mock_config.pop("model_config")  # pop a key that's required to pass
        expect_false = is_TorchModuleWrapper_config(mock_config)
        self.assertFalse(expect_false)

        mock_config.pop("checkpoint_path")  # pop the last key, mock_config is now empty
        expect_false = is_TorchModuleWrapper_config(mock_config)
        self.assertFalse(expect_false)

    def test_is_target_type(self) -> None:
        """Tests that is_target_type works as expected."""
        # Test data
        mock_config_test_utils_mock_type = DictConfig(
            {
                "_target_": f"{__name__}.TestUtilsTypeMockType",
            }
        )
        mock_config_test_utils_another_mock_type = DictConfig(
            {
                "_target_": f"{__name__}.TestUtilsTypeAnotherMockType",
            }
        )

        # Nominal cases
        expect_true = is_target_type(mock_config_test_utils_mock_type, TestUtilsTypeMockType)
        self.assertTrue(expect_true)
        expect_true = is_target_type(mock_config_test_utils_another_mock_type, TestUtilsTypeAnotherMockType)
        self.assertTrue(expect_true)

        # Failing cases
        expect_false = is_target_type(mock_config_test_utils_mock_type, TestUtilsTypeAnotherMockType)
        self.assertFalse(expect_false)
        expect_false = is_target_type(mock_config_test_utils_another_mock_type, TestUtilsTypeMockType)
        self.assertFalse(expect_false)

    def test_validate_type(self) -> None:
        """Tests that validate_type works as expected."""
        test_utils_type_mock_type = TestUtilsTypeMockType()

        # Nominal case
        validate_type(test_utils_type_mock_type, TestUtilsTypeMockType)

        # Failing case
        with self.assertRaises(AssertionError):
            validate_type(test_utils_type_mock_type, TestUtilsTypeAnotherMockType)

    def test_are_the_same_type(self) -> None:
        """Tests that are_the_same_type works as expected."""
        test_utils_type_mock_type = TestUtilsTypeMockType()
        another_test_utils_type_mock_type = TestUtilsTypeMockType()
        test_utils_type_another_mock_type = TestUtilsTypeAnotherMockType()

        # Nominal cases
        are_the_same_type(test_utils_type_mock_type, another_test_utils_type_mock_type)

        # Failing case
        with self.assertRaises(AssertionError):
            are_the_same_type(test_utils_type_mock_type, test_utils_type_another_mock_type)

    def test_validate_dict_type(self) -> None:
        """Tests that validate_dict_type works as expected."""
        # Nominal case
        mock_config = DictConfig(
            {
                "_convert_": "all",
                "correct_object": {
                    "_target_": f"{__name__}.TestUtilsTypeMockType",
                    "a": 1,
                    "b": 2.5,
                },
                "correct_object_2": {
                    "_target_": f"{__name__}.TestUtilsTypeMockType",
                    "a": 1,
                    "b": 2.5,
                },
            }
        )
        instantiated_config = hydra.utils.instantiate(mock_config)
        validate_dict_type(instantiated_config, TestUtilsTypeMockType)

        # Failing case
        mock_config.other_object = {
            "_target_": f"{__name__}.TestUtilsTypeAnotherMockType",
            "c": 1,
        }  # add another object type to mock_config, subsequent calls to function under test should fail
        instantiated_config = hydra.utils.instantiate(mock_config)
        with self.assertRaises(AssertionError):
            validate_dict_type(instantiated_config, TestUtilsTypeMockType)

    def test_find_builder_in_config(self) -> None:
        """Tests that find_builder_in_config works as expected."""
        # Nominal case
        mock_config = DictConfig(
            {
                "correct_object": {
                    "_target_": f"{__name__}.TestUtilsTypeMockType",
                    "a": 1,
                    "b": 2.5,
                },
                "other_object": {
                    "_target_": f"{__name__}.TestUtilsTypeAnotherMockType",
                    "c": 1,
                },
            }
        )
        test_utils_mock_type = find_builder_in_config(mock_config, TestUtilsTypeMockType)
        self.assertTrue(is_target_type(test_utils_mock_type, TestUtilsTypeMockType))

        test_utils_another_mock_type = find_builder_in_config(mock_config, TestUtilsTypeAnotherMockType)
        self.assertTrue(is_target_type(test_utils_another_mock_type, TestUtilsTypeAnotherMockType))

        # Failing cases
        del mock_config.other_object  # remove other_object from mock_config
        with self.assertRaises(ValueError):
            find_builder_in_config(mock_config, TestUtilsTypeAnotherMockType)  # will raise since it's not there anymore


if __name__ == '__main__':
    unittest.main()
