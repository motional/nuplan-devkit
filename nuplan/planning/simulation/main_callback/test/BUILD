load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_metric_file_callback",
    srcs = ["test_metric_file_callback.py"],
    deps = [
        "//nuplan/planning/simulation/main_callback:metric_file_callback",
    ],
)

py_test(
    name = "test_metric_aggregator_callback",
    srcs = ["test_metric_aggregator_callback.py"],
    deps = [
        "//nuplan/planning/metrics/aggregator/test:mock_abstract_metric_aggregator",
        "//nuplan/planning/simulation/main_callback:metric_aggregator_callback",
    ],
)

py_test(
    name = "test_publisher_callback",
    srcs = ["test_publisher_callback.py"],
    deps = [
        "//nuplan/planning/simulation/main_callback:publisher_callback",
    ],
)

py_test(
    name = "test_validation_callback",
    srcs = ["test_validation_callback.py"],
    deps = [
        "//nuplan/planning/simulation/main_callback:validation_callback",
    ],
)

py_test(
    name = "test_completion_callback",
    srcs = ["test_completion_callback.py"],
    deps = [
        "//nuplan/planning/simulation/main_callback:completion_callback",
    ],
)
