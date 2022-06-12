load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "metric_base",
    srcs = ["metric_base.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "violation_metric_base",
    srcs = ["violation_metric_base.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "within_bound_metric_base",
    srcs = ["within_bound_metric_base.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)
