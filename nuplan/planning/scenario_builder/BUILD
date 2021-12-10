load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_scenario_builder",
    srcs = ["abstract_scenario_builder.py"],
    deps = [
        "//nuplan/maps:abstract_map",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "abstract_scenario",
    srcs = ["abstract_scenario.py"],
    deps = [
        "//nuplan/actor_state:ego_state",
        "//nuplan/actor_state:state_representation",
        "//nuplan/actor_state:vehicle_parameters",
        "//nuplan/maps:abstract_map",
        "//nuplan/maps:maps_datatypes",
        "//nuplan/planning/simulation/observation:observation_type",
    ],
)

py_library(
    name = "scenario_filter",
    srcs = ["scenario_filter.py"],
)

py_library(
    name = "scenario_utils",
    srcs = ["scenario_utils.py"],
)
