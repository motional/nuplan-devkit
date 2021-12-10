load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "folder_builder",
    srcs = ["folder_builder.py"],
    deps = [
        "//nuplan/planning/nuboard/base:data_class",
    ],
)

py_library(
    name = "logging_builder",
    srcs = ["logging_builder.py"],
    deps = [
    ],
)

py_library(
    name = "metric_builder",
    srcs = ["metric_builder.py"],
    deps = [
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/metrics/evaluation_metrics/common:clearance_from_static_agents",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_distance_to_goal",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_is_comfortable",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lat_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lat_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_min_distance_to_track",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_rate",
        "//nuplan/planning/metrics/evaluation_metrics/common:speed_limit_violation",
        "//nuplan/planning/metrics/evaluation_metrics/common:time_gap_to_agent",
        "//nuplan/planning/metrics/evaluation_metrics/common:time_to_collision",
        "//nuplan/planning/scenario_builder:abstract_scenario",
    ],
)

py_library(
    name = "model_builder",
    srcs = ["model_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/modeling:nn_model",
    ],
)

py_library(
    name = "objectives_builder",
    srcs = ["objectives_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/modeling/objectives:abstract_objective",
    ],
)

py_library(
    name = "planner_builder",
    srcs = ["planner_builder.py"],
    deps = [
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner/ml_planner",
        "//nuplan/planning/training/modeling:planning_module",
    ],
)

py_library(
    name = "scenario_building_builder",
    srcs = ["scenario_building_builder.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders/utils:utils_type",
    ],
)

py_library(
    name = "scenario_filter_builder",
    srcs = ["scenario_filter_builder.py"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/script/builders/utils:utils_type",
    ],
)

py_library(
    name = "simulation_builder",
    srcs = ["simulation_builder.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/script/builders:metric_builder",
        "//nuplan/planning/script/builders:planner_builder",
        "//nuplan/planning/script/builders:scenario_filter_builder",
        "//nuplan/planning/simulation",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/callback:metric_callback",
        "//nuplan/planning/simulation/controller:abstract_controller",
        "//nuplan/planning/simulation/controller:log_playback",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:box",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/simulation_manager:abstract_simulation_manager",
        "//nuplan/planning/simulation/simulation_manager:step_simulation_manager",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "simulation_callback_builder",
    srcs = ["simulation_callback_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/callback:serialization_callback",
        "//nuplan/planning/simulation/callback:timing_callback",
    ],
)

py_library(
    name = "splitter_builder",
    srcs = ["splitter_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/data_loader:splitter",
    ],
)

py_library(
    name = "training_builder",
    srcs = ["training_builder.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/script/builders:objectives_builder",
        "//nuplan/planning/script/builders:scenario_filter_builder",
        "//nuplan/planning/script/builders:splitter_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders/utils:utils_checkpoint",
        "//nuplan/planning/training/callbacks:checkpoint_callback",
        "//nuplan/planning/training/callbacks:time_logging_callback",
        "//nuplan/planning/training/data_loader:datamodule",
        "//nuplan/planning/training/modeling:nn_model",
        "//nuplan/planning/training/modeling:planning_module",
        "//nuplan/planning/training/preprocessing:feature_caching_preprocessor",
        "//nuplan/planning/training/visualization:visualization_callbacks",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "training_metrics_builder",
    srcs = ["training_metrics_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/modeling/metrics:abstract_training_metric",
    ],
)

py_library(
    name = "worker_pool_builder",
    srcs = ["worker_pool_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/utils/multithreading:worker_parallel",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_ray",
        "//nuplan/planning/utils/multithreading:worker_sequential",
    ],
)
