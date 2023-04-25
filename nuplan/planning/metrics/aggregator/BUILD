load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "abstract_metric_aggregator",
    srcs = ["abstract_metric_aggregator.py"],
    deps = [
        "//nuplan/common/utils:io_utils",
        "//nuplan/planning/metrics:metric_dataframe",
        requirement("pyarrow"),
        requirement("s3fs"),
    ],
)

py_library(
    name = "weighted_average_metric_aggregator",
    srcs = ["weighted_average_metric_aggregator.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/metrics:metric_dataframe",
        "//nuplan/planning/metrics/aggregator:abstract_metric_aggregator",
    ],
)
