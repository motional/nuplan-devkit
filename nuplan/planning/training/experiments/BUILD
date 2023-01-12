load("@rules_python//python:defs.bzl", "py_library")

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
        "//nuplan/common/utils:file_backed_barrier",
        "//nuplan/common/utils:helpers",
        "//nuplan/common/utils:s3_utils",
        "//nuplan/database/common/blob_store:s3_store",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_filter_utils",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders:scenario_filter_builder",
        "//nuplan/planning/training/preprocessing:feature_preprocessor",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_sequential",
        "//nuplan/planning/utils/multithreading:worker_utils",
    ],
)

py_library(
    name = "cache_metadata_entry",
    srcs = ["cache_metadata_entry.py"],
    deps = [
        "//nuplan/database/common/blob_store:s3_store",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)
