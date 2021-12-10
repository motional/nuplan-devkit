load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_training_metric",
    srcs = ["abstract_training_metric.py"],
    deps = [
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "agents_imitation_metrics",
    srcs = ["agents_imitation_metrics.py"],
    deps = [
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/metrics:abstract_training_metric",
        "//nuplan/planning/training/preprocessing/features:agents_trajectories",
    ],
)

py_library(
    name = "planning_metrics",
    srcs = ["planning_metrics.py"],
    deps = [
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/metrics:abstract_training_metric",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)
