load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "helpers",
    srcs = [
        "helpers.py",
    ],
)

py_library(
    name = "s3_utils",
    srcs = ["s3_utils.py"],
    deps = [
        "//nuplan/common/utils:helpers",
        requirement("requests"),
        requirement("retry"),
        requirement("botocore"),
        requirement("aioboto3"),
        requirement("boto3"),
    ],
)

py_library(
    name = "io_utils",
    srcs = ["io_utils.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        requirement("aiofiles"),
    ],
)

py_library(
    name = "split_state",
    srcs = [
        "split_state.py",
    ],
)

py_library(
    name = "torch_math",
    srcs = ["torch_math.py"],
    deps = [
        requirement_torch("torch"),
    ],
)

py_library(
    name = "interpolatable_state",
    srcs = [
        "interpolatable_state.py",
    ],
    deps = [
        "//nuplan/common/utils:split_state",
    ],
)

py_library(
    name = "file_backed_barrier",
    srcs = [
        "file_backed_barrier.py",
    ],
    deps = [
        "//nuplan/common/utils:s3_utils",
        requirement("retry"),
    ],
)

py_library(
    name = "distributed_scenario_filter",
    srcs = [
        "distributed_scenario_filter.py",
    ],
    deps = [
        ":file_backed_barrier",
        ":helpers",
        ":s3_utils",
        "//nuplan/common/utils:io_utils",
        "//nuplan/database/common/blob_store:s3_store",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders:scenario_filter_builder",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_utils",
        requirement("pandas"),
        requirement("omegaconf"),
        requirement("s3fs"),
    ],
)
