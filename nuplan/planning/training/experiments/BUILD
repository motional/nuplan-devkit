load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "training",
    srcs = ["training.py"],
    deps = [
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:training_builder",
        "//nuplan/planning/training/callbacks:profile_callback",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "caching",
    srcs = ["caching.py"],
    deps = [
        ":cache_metadata_entry",
        "//nuplan/common/utils:distributed_scenario_filter",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders:scenario_filter_builder",
        "//nuplan/planning/training/preprocessing:feature_preprocessor",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_utils",
    ],
)

py_library(
    name = "cache_metadata_entry",
    srcs = ["cache_metadata_entry.py"],
    deps = [
        "//nuplan/common/utils:io_utils",
        "//nuplan/database/common/blob_store:s3_store",
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("s3fs"),
    ],
)
