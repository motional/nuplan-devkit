load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_planner",
    srcs = ["abstract_planner.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
        "//nuplan/planning/simulation/trajectory",
    ],
)

py_library(
    name = "end_to_end",
    srcs = ["end_to_end.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
        "//nuplan/planning/simulation/trajectory",
    ],
)

py_library(
    name = "log_future",
    srcs = ["log_future.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
        "//nuplan/planning/simulation/trajectory",
        "//nuplan/planning/simulation/trajectory:interpolated",
    ],
)

py_library(
    name = "simple_planner",
    srcs = ["simple_planner.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
        "//nuplan/planning/simulation/trajectory",
        "//nuplan/planning/simulation/trajectory:interpolated",
    ],
)
