load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "mock_abstract_metric_aggregator",
    srcs = ["mock_abstract_metric_aggregator.py"],
    deps = [
        "//nuplan/planning/metrics:metric_dataframe",
        "//nuplan/planning/metrics/aggregator:abstract_metric_aggregator",
        requirement("pandas"),
    ],
)

py_test(
    name = "test_weighted_average_metric_aggregator",
    srcs = ["test_weighted_average_metric_aggregator.py"],
    deps = [
        "//nuplan/planning/metrics:metric_dataframe",
        "//nuplan/planning/metrics/aggregator:weighted_average_metric_aggregator",
        requirement("pandas"),
    ],
)
