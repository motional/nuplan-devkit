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
        requirement("boto3"),
        requirement("requests"),
        requirement("retry"),
        requirement("botocore"),
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
