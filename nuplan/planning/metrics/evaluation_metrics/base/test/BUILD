load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_violation_metric_base",
    srcs = ["test_violation_metric_base.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/metrics/evaluation_metrics/base:violation_metric_base",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
    ],
)

py_test(
    name = "test_within_bound_metric_base",
    srcs = ["test_within_bound_metric_base.py"],
    deps = [
        "//nuplan/planning/metrics/evaluation_metrics/base:within_bound_metric_base",
    ],
)
