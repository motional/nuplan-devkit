load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "abstract_main_callback",
    srcs = ["abstract_main_callback.py"],
)

py_library(
    name = "multi_main_callback",
    srcs = ["multi_main_callback.py"],
    deps = [
        "//nuplan/planning/simulation/main_callback:abstract_main_callback",
    ],
)

py_library(
    name = "metric_file_callback",
    srcs = ["metric_file_callback.py"],
    deps = [
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/simulation/main_callback:abstract_main_callback",
    ],
)

py_library(
    name = "time_callback",
    srcs = ["time_callback.py"],
    deps = [
        "//nuplan/planning/simulation/main_callback:abstract_main_callback",
    ],
)

py_library(
    name = "metric_aggregator_callback",
    srcs = ["metric_aggregator_callback.py"],
    deps = [
        "//nuplan/planning/metrics:metric_dataframe",
        "//nuplan/planning/metrics/aggregator:abstract_metric_aggregator",
        "//nuplan/planning/simulation/main_callback:abstract_main_callback",
    ],
)

py_library(
    name = "publisher_callback",
    srcs = ["publisher_callback.py"],
    deps = [
        "//nuplan/planning/simulation/main_callback:abstract_main_callback",
    ],
)

py_library(
    name = "validation_callback",
    srcs = ["validation_callback.py"],
    deps = [
        "//nuplan/planning/simulation/main_callback:abstract_main_callback",
    ],
)

py_library(
    name = "completion_callback",
    srcs = ["completion_callback.py"],
    deps = [
        "//nuplan/planning/simulation/main_callback:abstract_main_callback",
    ],
)
