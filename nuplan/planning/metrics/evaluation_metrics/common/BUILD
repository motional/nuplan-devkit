load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "clearance_from_static_agents",
    srcs = ["clearance_from_static_agents.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/geometry:compute",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
    ],
)

py_library(
    name = "drivable_area_violation",
    srcs = ["drivable_area_violation.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:violation_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
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
    name = "ego_collisions",
    srcs = ["ego_collisions.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:violation_metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
    ],
)

py_library(
    name = "ego_distance_to_goal",
    srcs = ["ego_distance_to_goal.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
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
    name = "ego_min_distance_to_track",
    srcs = ["ego_min_distance_to_track.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
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
    name = "speed_limit_violation",
    srcs = ["speed_limit_violation.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:violation_metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:drivable_area_violation",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "time_gap_to_lead_agent",
    srcs = ["time_gap_to_lead_agent.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation/idm:utils",
    ],
)

py_library(
    name = "time_gap_to_rear_agent",
    srcs = ["time_gap_to_rear_agent.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation/idm:utils",
    ],
)

py_library(
    name = "time_to_collision",
    srcs = ["time_to_collision.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:state_extractors",
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
    name = "ego_progress_to_goal",
    srcs = ["ego_progress_to_goal.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
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
    name = "ego_relative_progress_to_goal",
    srcs = ["ego_relative_progress_to_goal.py"],
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
    name = "ego_safety_performance",
    srcs = ["ego_safety_performance.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:drivable_area_violation",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_min_distance_to_lead_agent",
        "//nuplan/planning/metrics/evaluation_metrics/common:time_to_collision",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
    ],
)

py_library(
    name = "distance_to_baseline",
    srcs = ["distance_to_baseline.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/utils:route_extractor",
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
    name = "ego_at_fault_collisions",
    srcs = ["ego_at_fault_collisions.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:violation_metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/idm:utils",
    ],
)

py_library(
    name = "ego_min_distance_to_lead_agent",
    srcs = ["ego_min_distance_to_lead_agent.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:metric_base",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation/idm:utils",
    ],
)
