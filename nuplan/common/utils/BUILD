load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

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
    ],
)

py_library(
    name = "split_state",
    srcs = [
        "split_state.py",
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
