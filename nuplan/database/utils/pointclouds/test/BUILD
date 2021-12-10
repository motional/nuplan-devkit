load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_lidar",
    size = "small",
    srcs = ["test_lidar.py"],
    deps = [
        "//nuplan/database/utils/pointclouds:lidar",
    ],
)
