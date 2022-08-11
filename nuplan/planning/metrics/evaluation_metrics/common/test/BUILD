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
    name = "test_drivable_area_violation",
    srcs = ["test_drivable_area_violation.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:drivable_area_violation",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
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
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/evaluation_metrics/common:speed_limit_violation",
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
    name = "test_ego_collisions_pedestrian",
    srcs = ["test_ego_collisions_pedestrian.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions_with_pedestrians",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_collisions_object",
    srcs = ["test_ego_collisions_object.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions_with_objects",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_collisions_bicycle",
    srcs = ["test_ego_collisions_bicycle.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions_with_bicycles",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_collisions_vehicle",
    srcs = ["test_ego_collisions_vehicle.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_at_fault_collisions_with_vehicles",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_lane_change",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)

py_test(
    name = "test_ego_is_making_progress",
    srcs = ["test_ego_is_making_progress.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_is_making_progress",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_progress_along_expert_route",
        "//nuplan/planning/metrics/utils:testing_utils",
    ],
)
