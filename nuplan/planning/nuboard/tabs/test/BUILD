load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_configuration_tab",
    size = "medium",
    srcs = ["test_configuration_tab.py"],
    deps = [
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/tabs:configuration_tab",
        "//nuplan/planning/nuboard/tabs:histogram_tab",
    ],
)

py_test(
    name = "test_histogram_tab",
    size = "medium",
    srcs = ["test_histogram_tab.py"],
    deps = [
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/tabs:histogram_tab",
    ],
)

py_test(
    name = "test_overview_tab",
    size = "medium",
    srcs = ["test_overview_tab.py"],
    deps = [
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/tabs:overview_tab",
    ],
)

py_test(
    name = "test_scenario_tab",
    size = "medium",
    srcs = ["test_scenario_tab.py"],
    deps = [
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/tabs:scenario_tab",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
    ],
)
