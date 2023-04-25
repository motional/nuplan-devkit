import functools
import os
import sys
from os.path import basename, dirname, isdir, join
from typing import Any, List, Optional

import pytest

NUPLAN_TEST_PLUGIN = "nuplan.common.utils.test_utils.plugin"


def parametrize_filebased(abspath: Optional[str], filename: str, relpath: Optional[str]) -> Any:
    """
    Converts a target json file as a source of parameters for pytest.
    :param abspath: Absolute path of the json file
    :param filename: Name of the json file
    :param relpath: Relative path to the json file
    :return A pytest parameter
    """
    if filename.endswith(".json"):
        id_ = filename[:-5]
        return pytest.param(
            None, id=id_, marks=[pytest.mark.nuplan_test(relpath=relpath, absdirpath=abspath, params=id_)]
        )
    else:
        return pytest.param(
            None, id="-", marks=[pytest.mark.nuplan_test(relpath=relpath, absdirpath=None, params=None)]
        )


def parametrize_dir(absdirpath: Optional[str], files: List[str], relpath: Optional[str]) -> List[Any]:
    """
    Converts a target json file as a source of parameters for pytest.
    :param absdirpath: Absolute path of the directory containing the json files
    :param files: Name of the json files
    :param relpath: Relative path to the json file
    :return A list of pytest parameters
    """
    parameters = [
        pytest.param(
            None,
            id="<newname>",
            marks=[pytest.mark.nuplan_test(relpath=relpath, absdirpath=absdirpath, params=None)],
        )
    ]
    for file in files:
        if file.endswith(".json"):
            parameters.append(parametrize_filebased(absdirpath, file, relpath))
    return parameters


def nuplan_test(path: Optional[str] = None) -> Any:
    """
    This decorator enable pytest to load a sample scene from a json file. The test can then be run normally with pytest
    if the plugin is added to PYTEST_PLUGINS. It can be replaced with any other framework for visual testing/debugging.
    """

    def impl_decorate(nuplan_test: Any) -> Any:
        if path is not None:
            name = sys.modules.get(nuplan_test.__module__).__file__  # type: ignore
            abspath = join(dirname(name), path)  # type: ignore

            if isdir(abspath):

                @functools.wraps(nuplan_test)
                @pytest.mark.usefixtures("scene")
                @pytest.mark.parametrize(
                    argnames="nuplan_test", argvalues=parametrize_dir(abspath, os.listdir(abspath), path)
                )
                def testwrapper(*args: Any, **kwargs: Any) -> Any:
                    return nuplan_test(*args, **kwargs)

                return testwrapper
            else:

                @functools.wraps(nuplan_test)
                @pytest.mark.usefixtures("scene")
                @pytest.mark.parametrize(
                    argnames="nuplan_test", argvalues=[parametrize_filebased(dirname(abspath), basename(abspath), path)]
                )
                def testwrapper(*args: Any, **kwargs: Any) -> Any:
                    return nuplan_test(*args, **kwargs)

                return testwrapper
        else:

            @functools.wraps(nuplan_test)
            @pytest.mark.nuplan_test(type="hardcoded", params=None, absdirpath=None, relpath=None)
            @pytest.mark.usefixtures("scene")
            @pytest.mark.parametrize(argnames="nuplan_test", argvalues=[None], ids=["-"])
            def testwrapper(*args: Any, **kwargs: Any) -> Any:
                return nuplan_test(*args, **kwargs)

            return testwrapper

    return impl_decorate
