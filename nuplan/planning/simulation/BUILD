load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "runner",
    srcs = ["runner.py"],
    deps = [
        "//nuplan/planning/simulation",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "simulation",
    srcs = ["simulation.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/planner:abstract_planner",
    ],
)

py_library(
    name = "simulation_setup",
    srcs = ["simulation_setup.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/controller:abstract_controller",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/simulation_manager:abstract_simulation_manager",
    ],
)
