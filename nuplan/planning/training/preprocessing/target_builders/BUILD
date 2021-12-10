load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_target_builder",
    srcs = ["abstract_target_builder.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
    ],
)

py_library(
    name = "ego_trajectory_target_builder",
    srcs = ["ego_trajectory_target_builder.py"],
    deps = [
        "@nuplan_devkit//nuplan/planning/scenario_builder:abstract_scenario",
        "@nuplan_devkit//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "@nuplan_devkit//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "@nuplan_devkit//nuplan/planning/training/preprocessing/features:trajectory",
        "@nuplan_devkit//nuplan/planning/training/preprocessing/features:trajectory_utils",
        "@nuplan_devkit//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
    ],
)
