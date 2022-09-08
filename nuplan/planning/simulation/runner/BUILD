load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "abstract_runner",
    srcs = ["abstract_runner.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/runner:runner_report",
    ],
)

py_library(
    name = "runner_report",
    srcs = ["runner_report.py"],
    deps = [
        "//nuplan/planning/simulation/planner:abstract_planner",
    ],
)

py_library(
    name = "executor",
    srcs = ["executor.py"],
    deps = [
        "//nuplan/planning/simulation/callback:metric_callback",
        "//nuplan/planning/simulation/callback:multi_callback",
        "//nuplan/planning/simulation/callback:simulation_log_callback",
        "//nuplan/planning/simulation/runner:abstract_runner",
        "//nuplan/planning/simulation/runner:runner_report",
        "//nuplan/planning/simulation/runner:simulations_runner",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "metric_runner",
    srcs = ["metric_runner.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation:simulation_log",
        "//nuplan/planning/simulation/callback:metric_callback",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/runner:abstract_runner",
        "//nuplan/planning/simulation/runner:runner_report",
    ],
)

py_library(
    name = "simulations_runner",
    srcs = ["simulations_runner.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/runner:abstract_runner",
        "//nuplan/planning/simulation/runner:runner_report",
    ],
)
