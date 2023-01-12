load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "ml_planner",
    srcs = ["ml_planner.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:planner_report",
        "//nuplan/planning/simulation/planner/ml_planner:model_loader",
        "//nuplan/planning/simulation/planner/ml_planner:transform_utils",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
    ],
)

py_library(
    name = "transform_utils",
    srcs = ["transform_utils.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:convert",
        "//nuplan/planning/metrics/utils:state_extractors",
    ],
)

py_library(
    name = "model_loader",
    srcs = ["model_loader.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
    ],
)
