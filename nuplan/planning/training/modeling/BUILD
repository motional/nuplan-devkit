load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nn_model",
    srcs = ["nn_model.py"],
    deps = [
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
    ],
)

py_library(
    name = "planning_module",
    srcs = ["planning_module.py"],
    deps = [
        "//nuplan/planning/training/modeling:nn_model",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/metrics:planning_metrics",
        "//nuplan/planning/training/modeling/objectives:abstract_objective",
        "//nuplan/planning/training/modeling/objectives:imitation_objective",
    ],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
    ],
)
