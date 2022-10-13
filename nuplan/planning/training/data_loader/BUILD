load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "datamodule",
    srcs = ["datamodule.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/data_loader:scenario_dataset",
        "//nuplan/planning/training/data_loader:splitter",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing:feature_collate",
        "//nuplan/planning/training/preprocessing:feature_preprocessor",
    ],
)

py_library(
    name = "log_splitter",
    srcs = ["log_splitter.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/data_loader:splitter",
    ],
)

py_library(
    name = "scenario_dataset",
    srcs = ["scenario_dataset.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing:feature_preprocessor",
    ],
)

py_library(
    name = "splitter",
    srcs = ["splitter.py"],
    deps = ["//nuplan/planning/scenario_builder:abstract_scenario"],
)

py_library(
    name = "distributed_sampler_wrapper",
    srcs = ["distributed_sampler_wrapper.py"],
    deps = [],
)
