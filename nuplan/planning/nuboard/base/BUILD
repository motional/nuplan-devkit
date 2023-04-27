load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "base_tab",
    srcs = ["base_tab.py"],
    deps = [
        "//nuplan/planning/metrics:metric_dataframe",
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/base:data_class",
    ],
)

py_library(
    name = "data_class",
    srcs = ["data_class.py"],
    deps = [
        "//nuplan/common/utils:io_utils",
        "//nuplan/planning/metrics:metric_dataframe",
    ],
)

py_library(
    name = "experiment_file_data",
    srcs = ["experiment_file_data.py"],
    deps = [
        "//nuplan/planning/metrics:metric_dataframe",
        "//nuplan/planning/nuboard/base:data_class",
    ],
)

py_library(
    name = "plot_data",
    srcs = ["plot_data.py"],
    deps = [
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/geometry:transform",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/simulation:simulation_log",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "@nuplan_devkit//nuplan/planning/utils/serialization:to_scene",
    ],
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
        "//nuplan/common/maps:abstract_map_factory",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/base:plot_data",
        "//nuplan/planning/nuboard/tabs/config:scenario_tab_config",
        requirement("selenium"),
    ],
)
