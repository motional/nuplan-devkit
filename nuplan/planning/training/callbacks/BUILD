load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "checkpoint_callback",
    srcs = ["checkpoint_callback.py"],
)

py_library(
    name = "profile_callback",
    srcs = ["profile_callback.py"],
    deps = [
    ],
)

py_library(
    name = "time_logging_callback",
    srcs = ["time_logging_callback.py"],
)

py_library(
    name = "validate_setup_callback",
    srcs = ["validate_setup_callback.py"],
    deps = [
        "//nuplan/planning/training/data_loader:datamodule",
        "//nuplan/planning/training/modeling:planning_module",
    ],
)
