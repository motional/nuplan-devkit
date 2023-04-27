load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_callback",
    srcs = ["abstract_callback.py"],
    deps = [
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
    ],
)

py_library(
    name = "metric_callback",
    srcs = ["metric_callback.py"],
    deps = [
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "serialization_callback",
    srcs = ["serialization_callback.py"],
    deps = [
        "//nuplan/common/actor_state:car_footprint",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/utils:io_utils",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/utils:color",
        "//nuplan/planning/utils/serialization:to_scene",
        requirement("ujson"),
        requirement("msgpack"),
    ],
)

py_library(
    name = "timing_callback",
    srcs = ["timing_callback.py"],
    deps = [
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
    ],
)

py_library(
    name = "visualization_callback",
    srcs = ["visualization_callback.py"],
    deps = [
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/visualization:abstract_visualization",
    ],
)

py_library(
    name = "multi_callback",
    srcs = ["multi_callback.py"],
    deps = [
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
    ],
)

py_library(
    name = "simulation_log_callback",
    srcs = ["simulation_log_callback.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation:simulation_log",
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)
