load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_predictor",
    srcs = ["abstract_predictor.py"],
    deps = [
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/predictor:predictor_report",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)

py_library(
    name = "log_future_predictor",
    srcs = ["log_future_predictor.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/predictor:abstract_predictor",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
    ],
)

py_library(
    name = "predictor_report",
    srcs = ["predictor_report.py"],
    deps = [
        requirement("numpy"),
    ],
)
