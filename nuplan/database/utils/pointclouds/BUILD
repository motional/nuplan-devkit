load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "pointcloud",
    srcs = ["pointcloud.py"],
)

py_library(
    name = "lidar",
    srcs = ["lidar.py"],
    deps = [
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils:plot",
        "//nuplan/database/utils/pointclouds:pointcloud",
    ],
)
