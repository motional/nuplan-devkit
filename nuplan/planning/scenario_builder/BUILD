load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_scenario_builder",
    srcs = ["abstract_scenario_builder.py"],
    deps = [
        "//nuplan/common/maps:abstract_map_factory",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "abstract_scenario",
    srcs = ["abstract_scenario.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
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
