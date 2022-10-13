load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_data_augmentation",
    srcs = ["abstract_data_augmentation.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "data_augmentation_util",
    srcs = ["data_augmentation_util.py"],
    deps = [
        requirement("casadi"),
    ],
)

py_library(
    name = "gaussian_smooth_agent_augmentation",
    srcs = ["gaussian_smooth_agent_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "kinematic_agent_augmentation",
    srcs = ["kinematic_agent_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "kinematic_history_agent_augmentation",
    srcs = ["kinematic_history_agent_augmentation.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "kinematic_history_generic_agent_augmentation",
    srcs = ["kinematic_history_generic_agent_augmentation.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "simple_agent_augmentation",
    srcs = ["simple_agent_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "agent_dropout_augmentation",
    srcs = ["agent_dropout_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "generic_agent_dropout_augmentation",
    srcs = ["generic_agent_dropout_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
        "//nuplan/planning/training/modeling:types",
    ],
)
