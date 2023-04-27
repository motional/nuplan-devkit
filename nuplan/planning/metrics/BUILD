load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "abstract_metric",
    srcs = ["abstract_metric.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "metric_engine",
    srcs = ["metric_engine.py"],
    deps = [
        "//nuplan/common/utils:io_utils",
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "metric_file",
    srcs = ["metric_file.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
    ],
)

py_library(
    name = "metric_result",
    srcs = ["metric_result.py"],
    deps = ["//nuplan/planning/metrics:metric_dataframe"],
)

py_library(
    name = "metric_dataframe",
    srcs = ["metric_dataframe.py"],
    deps = [
        "//nuplan/common/utils:io_utils",
        requirement("numpy"),
        requirement("pandas"),
        requirement("s3fs"),
    ],
)
