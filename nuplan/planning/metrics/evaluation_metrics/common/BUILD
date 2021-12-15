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
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:geometry",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
    ],
)

py_library(
    name = "drivable_area_violation",
    srcs = ["drivable_area_violation.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:metric_violation_aggregator",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
    ],
)

py_library(
    name = "ego_acceleration",
    srcs = ["ego_acceleration.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_collisions",
    srcs = ["ego_collisions.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:metric_violation_aggregator",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
    ],
)

py_library(
    name = "ego_distance_to_goal",
    srcs = ["ego_distance_to_goal.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_is_comfortable",
    srcs = ["ego_is_comfortable.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lat_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_rate",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_jerk",
    srcs = ["ego_jerk.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_lat_acceleration",
    srcs = ["ego_lat_acceleration.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_lat_jerk",
    srcs = ["ego_lat_jerk.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_lon_acceleration",
    srcs = ["ego_lon_acceleration.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_lon_jerk",
    srcs = ["ego_lon_jerk.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_min_distance_to_track",
    srcs = ["ego_min_distance_to_track.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
    ],
)

py_library(
    name = "ego_yaw_acceleration",
    srcs = ["ego_yaw_acceleration.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "ego_yaw_rate",
    srcs = ["ego_yaw_rate.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "speed_limit_violation",
    srcs = ["speed_limit_violation.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:metric_violation_aggregator",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_library(
    name = "time_gap_to_agent",
    srcs = ["time_gap_to_agent.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
    ],
)

py_library(
    name = "time_to_collision",
    srcs = ["time_to_collision.py"],
    deps = [
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)
