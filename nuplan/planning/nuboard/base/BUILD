load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "base_tab",
    srcs = ["base_tab.py"],
    deps = [
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/base:data_class",
    ],
)

py_library(
    name = "data_class",
    srcs = ["data_class.py"],
)

py_library(
    name = "simulation_tile",
    srcs = ["simulation_tile.py"],
    deps = [
        "//nuplan/actor_state:oriented_box",
        "//nuplan/actor_state:state_representation",
        "//nuplan/actor_state:transform_state",
        "//nuplan/actor_state:vehicle_parameters",
        "//nuplan/maps:abstract_map",
        "//nuplan/maps:abstract_map_objects",
        "//nuplan/maps:maps_datatypes",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
    ],
)
