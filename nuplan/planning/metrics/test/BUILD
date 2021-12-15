load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_metric_engine",
    size = "medium",
    srcs = ["test_metric_engine.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_distance_to_goal",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated",
    ],
)
