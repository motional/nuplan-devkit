load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "feature_collate",
    srcs = ["feature_collate.py"],
    deps = ["//nuplan/planning/training/modeling:types"],
)

py_library(
    name = "feature_preprocessor",
    srcs = ["feature_preprocessor.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
        "//nuplan/planning/training/preprocessing/utils:feature_cache",
        "//nuplan/planning/training/preprocessing/utils:utils_cache",
    ],
)
