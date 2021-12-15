load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_geometry",
    size = "medium",
    srcs = ["test_geometry.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/metrics/utils:geometry",
    ],
)

py_test(
    name = "test_metric_violation_aggregator",
    size = "medium",
    srcs = ["test_metric_violation_aggregator.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/utils:metric_violation_aggregator",
    ],
)
