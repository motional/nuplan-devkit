load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_simulation_manager",
    srcs = ["abstract_simulation_manager.py"],
    deps = ["//nuplan/planning/simulation/simulation_manager:simulation_iteration"],
)

py_library(
    name = "simulation_iteration",
    srcs = ["simulation_iteration.py"],
    deps = ["//nuplan/common/actor_state:state_representation"],
)

py_library(
    name = "step_simulation_manager",
    srcs = ["step_simulation_manager.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/simulation_manager:abstract_simulation_manager",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
    ],
)
