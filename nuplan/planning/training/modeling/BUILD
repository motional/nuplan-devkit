load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "torch_module_wrapper",
    srcs = ["torch_module_wrapper.py"],
    deps = [
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
    ],
)

py_library(
    name = "scriptable_torch_module_wrapper",
    srcs = ["scriptable_torch_module_wrapper.py"],
    deps = [
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
    ],
)

py_library(
    name = "lightning_module_wrapper",
    srcs = ["lightning_module_wrapper.py"],
    deps = [
        "//nuplan/planning/script/builders:lr_scheduler_builder",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
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
