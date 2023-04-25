load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "folder_builder",
    srcs = ["folder_builder.py"],
    deps = [
        "//nuplan/common/utils:io_utils",
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/nuboard/base:data_class",
    ],
)

py_library(
    name = "logging_builder",
    srcs = ["logging_builder.py"],
    deps = [
        requirement("tqdm"),
        requirement("omegaconf"),
    ],
)

py_library(
    name = "metric_builder",
    srcs = ["metric_builder.py"],
    deps = [
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/metrics/evaluation_metrics/common:drivable_area_compliance",
        "//nuplan/planning/metrics/evaluation_metrics/common:driving_direction_compliance",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_expert_l2_error",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_expert_l2_error_with_yaw",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_is_comfortable",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_is_making_progress",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lat_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lat_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_mean_speed",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_progress_along_expert_route",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_rate",
        "//nuplan/planning/metrics/evaluation_metrics/common:no_ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:planner_expert_average_heading_error_within_bound",
        "//nuplan/planning/metrics/evaluation_metrics/common:planner_expert_average_l2_error_within_bound",
        "//nuplan/planning/metrics/evaluation_metrics/common:planner_expert_final_heading_error_within_bound",
        "//nuplan/planning/metrics/evaluation_metrics/common:planner_expert_final_l2_error_within_bound",
        "//nuplan/planning/metrics/evaluation_metrics/common:planner_miss_rate_within_bound",
        "//nuplan/planning/metrics/evaluation_metrics/common:speed_limit_compliance",
        "//nuplan/planning/metrics/evaluation_metrics/common:time_to_collision_within_bound",
        "//nuplan/planning/scenario_builder:abstract_scenario",
    ],
)

py_library(
    name = "model_builder",
    srcs = ["model_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
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
        "//nuplan/planning/simulation/planner:log_future_planner",
        "//nuplan/planning/simulation/planner/ml_planner",
        "//nuplan/planning/training/modeling:lightning_module_wrapper",
    ],
)

py_library(
    name = "scenario_building_builder",
    srcs = ["scenario_building_builder.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders/utils:utils_type",
        requirement("pyogrio"),
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
        "//nuplan/common/utils:distributed_scenario_filter",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:metric_builder",
        "//nuplan/planning/script/builders:observation_builder",
        "//nuplan/planning/script/builders:planner_builder",
        "//nuplan/planning/script/builders:scenario_filter_builder",
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/callback:metric_callback",
        "//nuplan/planning/simulation/callback:multi_callback",
        "//nuplan/planning/simulation/controller:abstract_controller",
        "//nuplan/planning/simulation/controller:log_playback",
        "//nuplan/planning/simulation/controller:perfect_tracking",
        "//nuplan/planning/simulation/controller:two_stage_controller",
        "//nuplan/planning/simulation/controller/tracker:ilqr_tracker",
        "//nuplan/planning/simulation/controller/tracker:lqr",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:idm_agents",
        "//nuplan/planning/simulation/observation:lidar_pc",
        "//nuplan/planning/simulation/observation:tracks_observation",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/runner:simulations_runner",
        "//nuplan/planning/simulation/simulation_time_controller:abstract_simulation_time_controller",
        "//nuplan/planning/simulation/simulation_time_controller:step_simulation_time_controller",
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
        "//nuplan/planning/simulation/callback:simulation_log_callback",
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
    name = "scenario_builder",
    srcs = ["scenario_builder.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders:scenario_filter_builder",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_utils",
    ],
)

py_library(
    name = "training_builder",
    srcs = ["training_builder.py"],
    deps = [
        "//nuplan/planning/script/builders:data_augmentation_builder",
        "//nuplan/planning/script/builders:objectives_builder",
        "//nuplan/planning/script/builders:scenario_builder",
        "//nuplan/planning/script/builders:splitter_builder",
        "//nuplan/planning/script/builders:training_callback_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders/utils:utils_checkpoint",
        "//nuplan/planning/training/callbacks:checkpoint_callback",
        "//nuplan/planning/training/callbacks:scenario_scoring_callback",
        "//nuplan/planning/training/callbacks:time_logging_callback",
        "//nuplan/planning/training/callbacks:visualization_callback",
        "//nuplan/planning/training/data_loader:datamodule",
        "//nuplan/planning/training/modeling:lightning_module_wrapper",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/preprocessing:feature_preprocessor",
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement_torch("pytorch-lightning"),
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

py_library(
    name = "observation_builder",
    srcs = ["observation_builder.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/training/modeling:lightning_module_wrapper",
    ],
)

py_library(
    name = "metric_aggregator_builder",
    srcs = ["metric_aggregator_builder.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/metrics/aggregator:abstract_metric_aggregator",
        "//nuplan/planning/metrics/aggregator:weighted_average_metric_aggregator",
    ],
)

py_library(
    name = "main_callback_builder",
    srcs = ["main_callback_builder.py"],
    deps = [
        "//nuplan/planning/script/builders:metric_aggregator_builder",
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/simulation/main_callback:abstract_main_callback",
        "//nuplan/planning/simulation/main_callback:metric_aggregator_callback",
        "//nuplan/planning/simulation/main_callback:metric_file_callback",
        "//nuplan/planning/simulation/main_callback:metric_summary_callback",
        "//nuplan/planning/simulation/main_callback:multi_main_callback",
        "//nuplan/planning/simulation/main_callback:publisher_callback",
        "//nuplan/planning/simulation/main_callback:time_callback",
    ],
)

py_library(
    name = "simulation_log_builder",
    srcs = ["simulation_log_builder.py"],
    deps = [
        "//nuplan/planning/simulation:simulation_log",
    ],
)

py_library(
    name = "metric_runner_builder",
    srcs = ["metric_runner_builder.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/script/builders:metric_builder",
        "//nuplan/planning/simulation:simulation_log",
        "//nuplan/planning/simulation/callback:metric_callback",
        "//nuplan/planning/simulation/runner:metric_runner",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "data_augmentation_builder",
    srcs = ["data_augmentation_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
    ],
)

py_library(
    name = "training_callback_builder",
    srcs = ["training_callback_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
    ],
)

py_library(
    name = "lr_scheduler_builder",
    srcs = ["lr_scheduler_builder.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/utils/lr_schedulers:sequential_lr_scheduler",
    ],
)
