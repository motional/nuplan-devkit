load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_json",
    srcs = glob(["json/**/*.json"]),
    visibility = ["//visibility:private"],
)

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_clearance_from_static_agents",
    srcs = ["test_clearance_from_static_agents.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:clearance_from_static_agents",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_drivable_area_violation",
    srcs = ["test_drivable_area_violation.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:drivable_area_violation",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_acceleration",
    srcs = ["test_ego_acceleration.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_acceleration",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_collisions",
    srcs = ["test_ego_collisions.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_collisions",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_distance_to_goal",
    srcs = ["test_ego_distance_to_goal.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_distance_to_goal",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_is_comfortable",
    srcs = ["test_ego_is_comfortable.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_is_comfortable",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lat_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_jerk",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_rate",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_jerk",
    srcs = ["test_ego_jerk.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_lat_acceleration",
    srcs = ["test_ego_lat_acceleration.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lat_acceleration",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_lat_jerk",
    srcs = ["test_ego_lat_jerk.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lat_jerk",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_lon_acceleration",
    srcs = ["test_ego_lon_acceleration.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_acceleration",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_lon_jerk",
    srcs = ["test_ego_lon_jerk.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lon_jerk",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_min_distance_to_track",
    srcs = ["test_ego_min_distance_to_track.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_min_distance_to_track",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_yaw_acceleration",
    srcs = ["test_ego_yaw_acceleration.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_acceleration",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_yaw_rate",
    srcs = ["test_ego_yaw_rate.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_yaw_rate",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_speed_limit_violation",
    srcs = ["test_speed_limit_violation.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:drivable_area_violation",
        "//nuplan/planning/metrics/evaluation_metrics/common:speed_limit_violation",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_time_gap_to_lead_agent",
    srcs = ["test_time_gap_to_lead_agent.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:time_gap_to_lead_agent",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_time_gap_to_rear_agent",
    srcs = ["test_time_gap_to_rear_agent.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:time_gap_to_rear_agent",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_time_to_collision",
    srcs = ["test_time_to_collision.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/evaluation_metrics/common:time_to_collision",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_progress_to_goal",
    srcs = ["test_ego_progress_to_goal.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_progress_to_goal",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_relative_progress_to_goal",
    srcs = ["test_ego_relative_progress_to_goal.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_relative_progress_to_goal",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_mean_speed",
    srcs = ["test_ego_mean_speed.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_mean_speed",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_expert_l2_error",
    srcs = ["test_ego_expert_l2_error.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_expert_l2_error",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_expert_l2_error_with_yaw",
    srcs = ["test_ego_expert_l2_error_with_yaw.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_expert_l2_error_with_yaw",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_distance_to_baseline",
    srcs = ["test_distance_to_baseline.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:distance_to_baseline",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_progress_along_expert_route",
    srcs = ["test_ego_progress_along_expert_route.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_progress_along_expert_route",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_lane_change",
    srcs = ["test_ego_lane_change.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_at_fault_collisions",
    srcs = ["test_ego_at_fault_collisions.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_safety_performance",
    srcs = ["test_ego_safety_performance.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/common:drivable_area_violation",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_min_distance_to_lead_agent",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_safety_performance",
        "//nuplan/planning/metrics/evaluation_metrics/common:time_to_collision",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_min_distance_to_lead_agent",
    srcs = ["test_ego_min_distance_to_lead_agent.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_min_distance_to_lead_agent",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)
