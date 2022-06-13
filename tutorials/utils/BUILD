load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "tutorial_utils",
    srcs = ["tutorial_utils.py"],
    deps = [
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:simulation_tile",
        "//nuplan/planning/simulation/callback:serialization_callback",
        "//nuplan/planning/simulation/controller:abstract_controller",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/simulation_time_controller:abstract_simulation_time_controller",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
    ],
)
