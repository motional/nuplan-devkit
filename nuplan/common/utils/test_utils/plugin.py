from typing import Any, Generator

import pytest

from nuplan.common.utils.test_utils import instances


def pytest_configure(config: Any) -> None:
    """Configures pytest"""
    config.addinivalue_line("markers", "nuplan_test(type): mark test to run only on named environment")


def pytest_collection_finish(session: pytest.Session) -> None:
    """Collects the test session items"""
    for item in session.items:
        skip = item.get_closest_marker("skip")
        if skip is not None:
            continue

        marker = item.get_closest_marker("nuplan_test")
        if marker is None:
            continue

        instances.REGISTRY.add(
            item.nodeid, marker.kwargs["params"], marker.kwargs["absdirpath"], marker.kwargs["relpath"]
        )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Generator[Any, Any, Any]:
    """Fetches result from pytest"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture()
def scene(nuplan_test: Any, request: pytest.FixtureRequest) -> Generator[Any, Any, Any]:
    """Depending on the test type, skips, fails or yields the data for the test."""
    test_type = instances.REGISTRY.get_type(request.node.nodeid)
    if test_type == "invalid":
        pytest.fail("Invalid Test")
    elif test_type in ("hardcoded", "filebased"):
        yield instances.REGISTRY.get_data(request.node.nodeid)
    elif test_type == "newable":
        pytest.skip()
