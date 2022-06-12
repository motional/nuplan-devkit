import json
import os
from typing import Any, Dict, Optional


class TestInfo:
    """Stores different configurations for nuplan_test and retrieves the type of test."""

    def __init__(self, params: Optional[str], absdirpath: Optional[str], relpath: Optional[str]):
        """
        :param params: Parameters of the test
        :param absdirpath: Absolute path of the test json
        :param relpath: Relative path of the test json
        """
        self.params = params
        self.absdirpath = absdirpath
        self.relpath = relpath

    def is_hardcoded(self) -> bool:
        """
        Checks if the test is hardcoded
        :return: whether the test is hardcoded.
        """
        return self.params is None and self.absdirpath is None and self.relpath is None

    def is_invalid(self) -> bool:
        """
        Checks if the test configuration is invalid
        :return: whether the test configuration is invalid
        """
        return self.params is None and self.absdirpath is None and self.relpath is not None

    def is_newable(self) -> bool:
        """
        Checks if the test is newable
        :return: whether the test is newable.
        """
        return self.params is None and self.absdirpath is not None and self.relpath is not None

    def is_file_based(self) -> bool:
        """
        Checks if the test is based on a file
        :return: whether the test is based on a file.
        """
        return self.params is not None and self.absdirpath is not None and self.relpath is not None


class Registry:
    """Registry containing all the nuplan tests."""

    def __init__(self) -> None:
        """Initializes an empty registry"""
        self.registry: Dict[str, TestInfo] = {}

    def add(self, id_: str, params: Optional[str], absdirpath: Optional[str], relpath: Optional[str]) -> None:
        """Adds a test to the registry, fails if the same test is added twice.
        :param id_: The id of the test
        :param params: Parameters of the test
        :param absdirpath: Absolute path of the test json
        :param relpath: Relative path of the test json
        """
        if id_ not in self.registry:
            self.registry[id_] = TestInfo(params, absdirpath, relpath)
        else:
            raise RuntimeError("Tried to add the same node ID twice!")

    def get_type(self, id_: str) -> str:
        """
        Gets the configuration type of the queried test
        :param id_: Id of the test
        :return: String containing the type of test configuration
        """
        if self.registry[id_].is_invalid():
            return "invalid"
        if self.registry[id_].is_file_based():
            return "filebased"
        if self.registry[id_].is_hardcoded():
            return "hardcoded"
        if self.registry[id_].is_newable():
            return "newable"
        raise RuntimeError("Unknown test id: " + id_)

    def get_data(self, id_: str) -> Any:
        """
        Loads the information of the queried test from the registry from a json file
        :param id_: ID of the test
        :return: The test dict
        """
        if id_ in self.registry:
            test_info = self.registry[id_]
            if test_info.is_file_based():
                file_path = os.path.join(test_info.absdirpath, test_info.params) + ".json"  # type: ignore

                with open(file_path) as f:
                    return json.load(f)
        return {}


REGISTRY = Registry()
