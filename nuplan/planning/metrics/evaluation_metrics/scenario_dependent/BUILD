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
        "//nuplan/actor_state:state_representation",
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/maps:abstract_map",
        "//nuplan/maps:abstract_map_objects",
        "//nuplan/maps:maps_datatypes",
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
        "//nuplan/planning/simulation/observation/smart_agents/occupancy_map:strtree_occupancy_map",
        "//nuplan/planning/simulation/path:utils",
    ],
)
