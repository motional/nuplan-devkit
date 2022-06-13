load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_simulation_time_controller",
    srcs = ["abstract_simulation_time_controller.py"],
    deps = ["//nuplan/planning/simulation/simulation_time_controller:simulation_iteration"],
)

py_library(
    name = "simulation_iteration",
    srcs = ["simulation_iteration.py"],
    deps = ["//nuplan/common/actor_state:state_representation"],
)

py_library(
    name = "step_simulation_time_controller",
    srcs = ["step_simulation_time_controller.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/simulation_time_controller:abstract_simulation_time_controller",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)
