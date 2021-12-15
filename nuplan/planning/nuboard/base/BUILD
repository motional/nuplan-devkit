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
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:transform_state",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
    ],
)
