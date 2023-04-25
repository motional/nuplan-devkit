load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_mock_abstract_scenario",
    size = "small",
    srcs = ["test_mock_abstract_scenario.py"],
    deps = [
        "//nuplan/common/utils/test_utils:interface_validation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
    ],
)

py_test(
    name = "test_mock_abstract_scenario_builder",
    size = "small",
    srcs = ["test_mock_abstract_scenario_builder.py"],
    deps = [
        "//nuplan/common/utils/test_utils:interface_validation",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
    ],
)
