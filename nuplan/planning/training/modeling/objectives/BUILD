load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_objective",
    srcs = ["abstract_objective.py"],
    deps = [
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "scenario_weight_utils",
    srcs = ["scenario_weight_utils.py"],
    deps = [
    ],
)

py_library(
    name = "agents_imitation_objective",
    srcs = ["agents_imitation_objective.py"],
    deps = [
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/objectives:abstract_objective",
        "//nuplan/planning/training/preprocessing/features:agents_trajectories",
    ],
)

py_library(
    name = "imitation_objective",
    srcs = ["imitation_objective.py"],
    deps = [
        ":scenario_weight_utils",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/objectives:abstract_objective",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)

py_library(
    name = "trajectory_weight_decay_imitation_objective",
    srcs = ["trajectory_weight_decay_imitation_objective.py"],
    deps = [
        ":scenario_weight_utils",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/objectives:abstract_objective",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)
