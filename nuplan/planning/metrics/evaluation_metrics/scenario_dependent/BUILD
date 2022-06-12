load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "ego_stop_at_stop_line",
    srcs = ["ego_stop_at_stop_line.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:scene_object",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/database/utils:geometry",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:violation_metric_base",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/idm:utils",
        "//nuplan/planning/simulation/occupancy_map:strtree_occupancy_map",
        "//nuplan/planning/simulation/path:utils",
    ],
)
