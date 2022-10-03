load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "drivable_area_compliance",
    srcs = ["drivable_area_compliance.py"],
    deps = [
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        requirement("sympy"),
    ],
)

py_library(
    name = "ego_acceleration",
    srcs = ["ego_acceleration.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:within_bound_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_is_comfortable",
    srcs = ["ego_is_comfortable.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lat_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_rate",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_jerk",
    srcs = ["ego_jerk.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:within_bound_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_lat_acceleration",
    srcs = ["ego_lat_acceleration.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:within_bound_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_lat_jerk",
    srcs = ["ego_lat_jerk.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:within_bound_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_lon_acceleration",
    srcs = ["ego_lon_acceleration.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:within_bound_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_lon_jerk",
    srcs = ["ego_lon_jerk.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:within_bound_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_yaw_acceleration",
    srcs = ["ego_yaw_acceleration.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:within_bound_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_yaw_rate",
    srcs = ["ego_yaw_rate.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:within_bound_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "speed_limit_compliance",
    srcs = ["speed_limit_compliance.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:violation_metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "time_to_collision_within_bound",
    srcs = ["time_to_collision_within_bound.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/evaluation_metrics/common:no_ego_at_fault_collisions",
        "//nuplan/planning/metrics/utils:collision_utils",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation/idm:utils",
    ],
)

py_library(
    name = "ego_expert_l2_error",
    srcs = ["ego_expert_l2_error.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:expert_comparisons",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_expert_l2_error_with_yaw",
    srcs = ["ego_expert_l2_error_with_yaw.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:expert_comparisons",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_progress_along_expert_route",
    srcs = ["ego_progress_along_expert_route.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:route_extractor",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_mean_speed",
    srcs = ["ego_mean_speed.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_lane_change",
    srcs = ["ego_lane_change.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:route_extractor",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "no_ego_at_fault_collisions",
    srcs = ["no_ego_at_fault_collisions.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:collision_utils",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/idm:utils",
    ],
)

py_library(
    name = "ego_is_making_progress",
    srcs = ["ego_is_making_progress.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_progress_along_expert_route",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "planner_expert_average_l2_error_within_bound",
    srcs = ["planner_expert_average_l2_error_within_bound.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:expert_comparisons",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
    ],
)

py_library(
    name = "planner_expert_final_l2_error_within_bound",
    srcs = ["planner_expert_final_l2_error_within_bound.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:planner_expert_average_l2_error_within_bound",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "planner_miss_rate_within_bound",
    srcs = ["planner_miss_rate_within_bound.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:planner_expert_average_l2_error_within_bound",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "planner_expert_average_heading_error_within_bound",
    srcs = ["planner_expert_average_heading_error_within_bound.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:planner_expert_average_l2_error_within_bound",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "planner_expert_final_heading_error_within_bound",
    srcs = ["planner_expert_final_heading_error_within_bound.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:planner_expert_average_l2_error_within_bound",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "driving_direction_compliance",
    srcs = ["driving_direction_compliance.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:route_extractor",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)
