load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

filegroup(
    name = "test_json",
    srcs = glob(["json/**/*.json"]),
    visibility = ["//visibility:private"],
)

py_test(
    name = "test_ego_stop_at_stop_line",
    size = "small",
    srcs = ["test_ego_stop_at_stop_line.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/scenario_dependent:ego_stop_at_stop_line",
        "//nuplan/planning/metrics/utils:testing_utils",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
    ],
)
